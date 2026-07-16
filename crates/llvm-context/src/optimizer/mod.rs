//! The LLVM optimizing tools.

use serde::Deserialize;
use serde::Serialize;

use crate::polkavm::context::attribute::Attribute;
use crate::target_machine::TargetMachine;

use self::settings::size_level::SizeLevel;
use self::settings::Settings;

pub mod settings;

/// The LLVM optimizing tools.
#[derive(Debug, Serialize, Deserialize)]
pub struct Optimizer {
    /// The optimizer settings.
    settings: Settings,
    /// Whether the module being optimized was produced by the newyork IR
    /// generator. The aggressive code-size pipeline is specific to newyork's
    /// outlined code; the stock Yul path must not run it (see [`Optimizer::run`]).
    #[serde(skip)]
    newyork: bool,
}

impl Optimizer {
    /// A shortcut constructor.
    pub fn new(settings: Settings) -> Self {
        Self {
            settings,
            newyork: false,
        }
    }

    /// Marks this optimizer as operating on a module produced by the newyork IR
    /// generator, enabling the aggressive code-size pipeline in [`Optimizer::run`].
    pub fn enable_newyork_pipeline(&mut self) {
        self.newyork = true;
    }

    /// Whether the module being optimized was produced by the newyork IR generator.
    /// Gates the two pieces of newyork-specific codegen that the stock Yul path must not use:
    /// the aggressive code-size pipeline in [`Optimizer::run`] and the `+unaligned-scalar-mem`
    /// target feature in [`crate::target_machine::TargetMachine`].
    pub fn is_newyork(&self) -> bool {
        self.newyork
    }

    /// Runs the new pass manager.
    ///
    /// For modules produced by the **newyork** IR generator a code-size pipeline runs:
    ///
    /// 1. `default<O{level}>`: initial optimization at the configured size level.
    /// 2. `ipsccp,deadargelim`: propagate inter-function constants through outlined helpers and
    ///    remove now-constant arguments.
    /// 3. `attributor`: infer function attributes the next stage can exploit.
    /// 4. `default<O1>`: re-optimize with newly discovered constants and attributes.
    ///
    /// `mergefunc` runs before stage 1, after stage 1, and after stage 4 to deduplicate functions
    /// that become identical as the surrounding passes canonicalize them.
    /// The second `default` pass is hard-coded to `O1` rather than the configured size level: an
    /// `Oz` second pass reproducibly changes the post-link PVM blob for some contracts in a way
    /// that fails pallet-revive deployment ("Code upload failed: Timeout"). The exact pass has not
    /// been isolated, so `O1` is kept as the empirically safe choice.
    ///
    /// This pipeline is specific to newyork's outlined code. The stock Yul path uses a plain
    /// `default<O{level}>`, matching upstream: its IR does not benefit from the extra inter-
    /// procedural passes, and `attributor`'s inference is unsound on it (the `attributor`+`O1`
    /// combination miscompiles, e.g. `complex/array_one_element` returns the callee selector
    /// instead of the external-call result). Gating on the IR generator keeps each path on the
    /// pipeline it was validated against.
    pub fn run(
        &self,
        target_machine: &TargetMachine,
        module: &inkwell::module::Module,
    ) -> Result<(), inkwell::support::LLVMString> {
        self.apply_size_optimization_attributes(module);

        let optimization_level = self.settings.middle_end_pipeline_level();
        let pass_pipeline = if self.newyork {
            format!(
                "mergefunc,default<O{optimization_level}>,mergefunc,ipsccp,deadargelim,attributor,default<O1>,mergefunc"
            )
        } else {
            format!("default<O{optimization_level}>")
        };
        target_machine.run_optimization_passes(module, &pass_pipeline)
    }

    /// Expresses the configured size-optimization level as function attributes.
    ///
    /// LLVM 23 removed the `Os`/`Oz` textual pass-pipeline levels; size
    /// optimization is now driven by the `optsize`/`minsize` function
    /// attributes on top of an `O2` pipeline (see
    /// [`Settings::middle_end_pipeline_level`]). This mirrors how clang lowers
    /// `-Os`/`-Oz`: `-Os` sets `optsize`, `-Oz` sets both `optsize` and
    /// `minsize`. The attributes are attached to every function definition in
    /// the module before the pipeline runs; declarations are left untouched.
    fn apply_size_optimization_attributes(&self, module: &inkwell::module::Module) {
        let size_attributes: &[Attribute] = match self.settings.level_middle_end_size {
            SizeLevel::Zero => &[],
            SizeLevel::S => &[Attribute::OptimizeForSize],
            SizeLevel::Z => &[Attribute::OptimizeForSize, Attribute::MinSize],
        };
        if size_attributes.is_empty() {
            return;
        }

        let context = module.get_context();
        for attribute in size_attributes {
            let llvm_attribute = context.create_enum_attribute(*attribute as u32, 0);
            let mut current_function = module.get_first_function();
            while let Some(function) = current_function {
                if function.count_basic_blocks() > 0 {
                    function
                        .add_attribute(inkwell::attributes::AttributeLoc::Function, llvm_attribute);
                }
                current_function = function.get_next_function();
            }
        }
    }

    /// Returns the optimizer settings reference.
    pub fn settings(&self) -> &Settings {
        &self.settings
    }
}
