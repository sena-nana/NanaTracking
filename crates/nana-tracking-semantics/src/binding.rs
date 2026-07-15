use std::{collections::BTreeMap, fmt};

use nana_tracking_protocol::{NanaTrackingDescriptor, NanaTrackingResult, SignalId, SignalState};
use serde::{Deserialize, Serialize};

use crate::{DerivedSample, SemanticFrame, SemanticId};

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ModelParameterId(String);

impl ModelParameterId {
    /// Creates a validated model parameter identifier.
    ///
    /// # Errors
    ///
    /// Returns [`BindingError::InvalidTarget`] for empty or control-character-containing names.
    pub fn new(value: impl Into<String>) -> Result<Self, BindingError> {
        let value = value.into();
        if value.trim().is_empty() || value.chars().any(char::is_control) {
            return Err(BindingError::InvalidTarget);
        }
        Ok(Self(value))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    fn validate(&self) -> Result<(), BindingError> {
        if self.0.trim().is_empty() || self.0.chars().any(char::is_control) {
            Err(BindingError::InvalidTarget)
        } else {
            Ok(())
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SignalExpression {
    Ntp(SignalId),
    Semantic(SemanticId),
    Positive(Box<Self>),
    Negative(Box<Self>),
    Average(Vec<Self>),
    Sum(Vec<Self>),
    Maximum(Vec<Self>),
    Constant(f32),
}

impl SignalExpression {
    #[must_use]
    pub fn positive(source: Self) -> Self {
        Self::Positive(Box::new(source))
    }

    #[must_use]
    pub fn negative(source: Self) -> Self {
        Self::Negative(Box::new(source))
    }

    fn evaluate(
        &self,
        ntp: &NanaTrackingResult,
        semantics: &SemanticFrame,
    ) -> Result<Option<DerivedSample>, BindingError> {
        match self {
            Self::Ntp(id) => Ok(ntp.rig.get(*id).and_then(|sample| {
                sample.value.map(|value| DerivedSample {
                    value,
                    confidence: sample.confidence,
                    state: sample.state,
                    sample_capture_timestamp_ns: sample.sample_capture_timestamp_ns,
                    sample_age_ns: semantics
                        .evaluation_timestamp_ns
                        .saturating_sub(sample.sample_capture_timestamp_ns),
                })
            })),
            Self::Semantic(id) => Ok(semantics.get(*id).copied()),
            Self::Positive(source) => Ok(source.evaluate(ntp, semantics)?.map(|mut sample| {
                sample.value = sample.value.max(0.0);
                sample
            })),
            Self::Negative(source) => Ok(source.evaluate(ntp, semantics)?.map(|mut sample| {
                sample.value = (-sample.value).max(0.0);
                sample
            })),
            Self::Average(sources) => {
                combine_expression(sources, ntp, semantics, CombineMode::Average)
            }
            Self::Sum(sources) => combine_expression(sources, ntp, semantics, CombineMode::Add),
            Self::Maximum(sources) => {
                combine_expression(sources, ntp, semantics, CombineMode::Maximum)
            }
            Self::Constant(value) if value.is_finite() => Ok(Some(DerivedSample {
                value: *value,
                confidence: 1.0,
                state: SignalState::Observed,
                sample_capture_timestamp_ns: ntp.capture_timestamp_ns,
                sample_age_ns: semantics
                    .evaluation_timestamp_ns
                    .saturating_sub(ntp.capture_timestamp_ns),
            })),
            Self::Constant(_) => Err(BindingError::InvalidExpression),
        }
    }

    fn validate(&self) -> Result<(), BindingError> {
        match self {
            Self::Ntp(_) | Self::Semantic(_) => Ok(()),
            Self::Positive(source) | Self::Negative(source) => source.validate(),
            Self::Average(sources) | Self::Sum(sources) | Self::Maximum(sources) => {
                if sources.is_empty() {
                    return Err(BindingError::InvalidExpression);
                }
                sources.iter().try_for_each(Self::validate)
            }
            Self::Constant(value) if value.is_finite() => Ok(()),
            Self::Constant(_) => Err(BindingError::InvalidExpression),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum BindingCurve {
    Linear,
    Smoothstep,
    Power(f32),
    PiecewiseLinear(Vec<CurvePoint>),
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CurvePoint {
    pub input: f32,
    pub output: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BindingTransform {
    pub curve: BindingCurve,
    pub deadzone: f32,
    pub clamp_min: f32,
    pub clamp_max: f32,
    pub scale: f32,
    pub offset: f32,
    pub invert: bool,
}

impl Default for BindingTransform {
    fn default() -> Self {
        Self {
            curve: BindingCurve::Linear,
            deadzone: 0.0,
            clamp_min: 0.0,
            clamp_max: 1.0,
            scale: 1.0,
            offset: 0.0,
            invert: false,
        }
    }
}

impl BindingTransform {
    fn validate(&self) -> Result<(), BindingError> {
        if ![
            self.deadzone,
            self.clamp_min,
            self.clamp_max,
            self.scale,
            self.offset,
        ]
        .into_iter()
        .all(f32::is_finite)
            || !(0.0..1.0).contains(&self.deadzone)
            || self.clamp_min > self.clamp_max
        {
            return Err(BindingError::InvalidTransform);
        }
        match &self.curve {
            BindingCurve::Linear | BindingCurve::Smoothstep => Ok(()),
            BindingCurve::Power(exponent) if exponent.is_finite() && *exponent > 0.0 => Ok(()),
            BindingCurve::PiecewiseLinear(points)
                if points.len() >= 2
                    && points
                        .iter()
                        .all(|point| point.input.is_finite() && point.output.is_finite())
                    && points.windows(2).all(|pair| pair[0].input < pair[1].input) =>
            {
                Ok(())
            }
            _ => Err(BindingError::InvalidTransform),
        }
    }

    fn apply(&self, value: f32) -> f32 {
        let mut value = if value.abs() <= self.deadzone {
            0.0
        } else {
            value.signum() * (value.abs() - self.deadzone) / (1.0 - self.deadzone)
        };
        value = match &self.curve {
            BindingCurve::Linear => value,
            BindingCurve::Smoothstep => {
                let sign = value.signum();
                let magnitude = value.abs().clamp(0.0, 1.0);
                sign * magnitude * magnitude * (3.0 - 2.0 * magnitude)
            }
            BindingCurve::Power(exponent) => value.signum() * value.abs().powf(*exponent),
            BindingCurve::PiecewiseLinear(points) => piecewise(value, points),
        };
        value *= self.scale;
        if self.invert {
            value = -value;
        }
        (value + self.offset).clamp(self.clamp_min, self.clamp_max)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum CombineMode {
    Replace,
    Add,
    Average,
    Maximum,
    Minimum,
}

/// Declarative schema requested by NTP consumers. Layer ownership is intentionally external.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RigBinding {
    pub source: SignalExpression,
    pub target: ModelParameterId,
    pub transform: BindingTransform,
    pub combine: CombineMode,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum BindingLayer {
    Orthogonal,
    Compatibility,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LayeredBinding {
    pub layer: BindingLayer,
    pub binding: RigBinding,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct SignalRequirements {
    pub required_signals: Vec<SignalId>,
    pub preferred_signals: Vec<SignalId>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequirementResolution {
    pub missing_required: Vec<SignalId>,
    pub missing_preferred: Vec<SignalId>,
}

impl RequirementResolution {
    #[must_use]
    pub fn is_compatible(&self) -> bool {
        self.missing_required.is_empty()
    }
}

impl SignalRequirements {
    #[must_use]
    pub fn resolve(&self, descriptor: &NanaTrackingDescriptor) -> RequirementResolution {
        RequirementResolution {
            missing_required: self
                .required_signals
                .iter()
                .copied()
                .filter(|id| !descriptor.supported_signals.contains(*id))
                .collect(),
            missing_preferred: self
                .preferred_signals
                .iter()
                .copied()
                .filter(|id| !descriptor.supported_signals.contains(*id))
                .collect(),
        }
    }

    fn validate(&self) -> Result<(), BindingError> {
        let mut all = self.required_signals.clone();
        all.extend(self.preferred_signals.iter().copied());
        all.sort_unstable();
        if all.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(BindingError::DuplicateRequirement);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BindingProfile {
    pub name: String,
    pub requirements: SignalRequirements,
    pub bindings: Vec<LayeredBinding>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelParameterValue {
    pub value: f32,
    pub confidence: f32,
    pub state: SignalState,
    pub sample_age_ns: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum BindingError {
    InvalidTarget,
    InvalidExpression,
    InvalidTransform,
    DuplicateRequirement,
    DuplicateBinding(ModelParameterId),
    LayerConflict(ModelParameterId),
    CombineConflict(ModelParameterId),
    FrameMismatch,
}

impl fmt::Display for BindingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTarget => formatter.write_str("invalid model parameter target"),
            Self::InvalidExpression => formatter.write_str("invalid signal expression"),
            Self::InvalidTransform => formatter.write_str("invalid binding transform"),
            Self::DuplicateRequirement => formatter.write_str("signal requirement is duplicated"),
            Self::DuplicateBinding(target) => {
                write!(formatter, "duplicate binding for {}", target.as_str())
            }
            Self::LayerConflict(target) => write!(
                formatter,
                "orthogonal and compatibility layers both drive {}",
                target.as_str()
            ),
            Self::CombineConflict(target) => write!(
                formatter,
                "incompatible combine rules for {}",
                target.as_str()
            ),
            Self::FrameMismatch => {
                formatter.write_str("NTP and semantic frames do not describe the same sample")
            }
        }
    }
}

impl std::error::Error for BindingError {}

pub struct BindingEvaluator {
    profile: BindingProfile,
    targets: Vec<ModelParameterId>,
    target_indices: Vec<usize>,
}

#[derive(Clone, Copy, Debug)]
struct Accumulator {
    sample: DerivedSample,
    count: u16,
    mode: CombineMode,
}

/// Caller-owned scratch space for allocation-free repeated binding evaluation.
///
/// Create it with [`BindingEvaluator::new_buffer`] after compiling a profile. Reusing the same
/// buffer preserves its storage across frames; changing evaluators transparently resizes it once.
#[derive(Debug, Default)]
pub struct BindingEvaluationBuffer {
    accumulators: Vec<Option<Accumulator>>,
    values: Vec<Option<ModelParameterValue>>,
}

impl BindingEvaluator {
    /// Validates and compiles a binding profile.
    ///
    /// # Errors
    ///
    /// Returns a validation error for malformed transforms/expressions, duplicate target sources,
    /// incompatible combine modes, or cross-layer target ownership.
    pub fn new(profile: BindingProfile) -> Result<Self, BindingError> {
        profile.requirements.validate()?;
        let mut owners: BTreeMap<
            &ModelParameterId,
            (BindingLayer, CombineMode, Vec<&SignalExpression>),
        > = BTreeMap::new();
        for layered in &profile.bindings {
            layered.binding.target.validate()?;
            layered.binding.source.validate()?;
            layered.binding.transform.validate()?;
            match owners.get_mut(&layered.binding.target) {
                None => {
                    owners.insert(
                        &layered.binding.target,
                        (
                            layered.layer,
                            layered.binding.combine,
                            vec![&layered.binding.source],
                        ),
                    );
                }
                Some((layer, combine, sources)) => {
                    if *layer != layered.layer {
                        return Err(BindingError::LayerConflict(layered.binding.target.clone()));
                    }
                    if *combine != layered.binding.combine || *combine == CombineMode::Replace {
                        return Err(BindingError::CombineConflict(
                            layered.binding.target.clone(),
                        ));
                    }
                    if sources.contains(&&layered.binding.source) {
                        return Err(BindingError::DuplicateBinding(
                            layered.binding.target.clone(),
                        ));
                    }
                    sources.push(&layered.binding.source);
                }
            }
        }
        let mut targets = profile
            .bindings
            .iter()
            .map(|layered| layered.binding.target.clone())
            .collect::<Vec<_>>();
        targets.sort();
        targets.dedup();
        let target_indices = profile
            .bindings
            .iter()
            .map(|layered| targets.partition_point(|candidate| candidate < &layered.binding.target))
            .collect();
        Ok(Self {
            profile,
            targets,
            target_indices,
        })
    }

    #[must_use]
    pub fn profile(&self) -> &BindingProfile {
        &self.profile
    }

    #[must_use]
    pub fn targets(&self) -> &[ModelParameterId] {
        &self.targets
    }

    #[must_use]
    pub fn new_buffer(&self) -> BindingEvaluationBuffer {
        BindingEvaluationBuffer {
            accumulators: vec![None; self.targets.len()],
            values: vec![None; self.targets.len()],
        }
    }

    /// Evaluates all available bindings against one NTP and semantic frame pair.
    ///
    /// # Errors
    ///
    /// Returns [`BindingError::FrameMismatch`] for unrelated input frames or
    /// [`BindingError::InvalidExpression`] if a runtime constant is non-finite.
    pub fn evaluate(
        &self,
        ntp: &NanaTrackingResult,
        semantics: &SemanticFrame,
    ) -> Result<BTreeMap<ModelParameterId, ModelParameterValue>, BindingError> {
        let mut buffer = self.new_buffer();
        let values = self.evaluate_buffered(ntp, semantics, &mut buffer)?;
        Ok(self
            .targets
            .iter()
            .cloned()
            .zip(values.iter().copied())
            .filter_map(|(target, value)| value.map(|value| (target, value)))
            .collect())
    }

    /// Evaluates into caller-owned storage without allocating after the buffer reaches the
    /// evaluator's target count. Output slots have the same stable order as [`Self::targets`].
    /// Missing source data is represented by `None` for only that target.
    ///
    /// # Errors
    ///
    /// Returns [`BindingError::FrameMismatch`] for unrelated input frames or
    /// [`BindingError::InvalidExpression`] if a runtime constant is non-finite.
    pub fn evaluate_buffered<'buffer>(
        &self,
        ntp: &NanaTrackingResult,
        semantics: &SemanticFrame,
        buffer: &'buffer mut BindingEvaluationBuffer,
    ) -> Result<&'buffer [Option<ModelParameterValue>], BindingError> {
        if ntp.session_id != semantics.session_id
            || ntp.generation != semantics.generation
            || ntp.sequence != semantics.sequence
            || ntp.capture_timestamp_ns != semantics.capture_timestamp_ns
        {
            return Err(BindingError::FrameMismatch);
        }
        if buffer.accumulators.len() != self.targets.len() {
            buffer.accumulators.resize(self.targets.len(), None);
            buffer.values.resize(self.targets.len(), None);
        }
        buffer.accumulators.fill(None);
        buffer.values.fill(None);
        for (layered, target_index) in self.profile.bindings.iter().zip(&self.target_indices) {
            let Some(mut sample) = layered.binding.source.evaluate(ntp, semantics)? else {
                continue;
            };
            sample.value = layered.binding.transform.apply(sample.value);
            match &mut buffer.accumulators[*target_index] {
                None => {
                    buffer.accumulators[*target_index] = Some(Accumulator {
                        sample,
                        count: 1,
                        mode: layered.binding.combine,
                    });
                }
                Some(accumulator) => {
                    accumulator.sample =
                        combine_samples(accumulator.sample, sample, accumulator.mode);
                    accumulator.count += 1;
                }
            }
        }
        for (value, accumulator) in buffer.values.iter_mut().zip(&buffer.accumulators) {
            let Some(mut accumulator) = *accumulator else {
                continue;
            };
            if accumulator.mode == CombineMode::Average {
                accumulator.sample.value /= f32::from(accumulator.count);
            }
            *value = Some(ModelParameterValue {
                value: accumulator.sample.value,
                confidence: accumulator.sample.confidence,
                state: accumulator.sample.state,
                sample_age_ns: accumulator.sample.sample_age_ns,
            });
        }
        Ok(&buffer.values)
    }
}

fn combine_expression(
    sources: &[SignalExpression],
    ntp: &NanaTrackingResult,
    semantics: &SemanticFrame,
    mode: CombineMode,
) -> Result<Option<DerivedSample>, BindingError> {
    let mut combined = None;
    let mut count = 0_u16;
    for source in sources {
        let Some(sample) = source.evaluate(ntp, semantics)? else {
            return Ok(None);
        };
        combined = Some(combined.map_or(sample, |current| combine_samples(current, sample, mode)));
        count += 1;
    }
    if mode == CombineMode::Average {
        if let Some(sample) = combined.as_mut() {
            sample.value /= f32::from(count);
        }
    }
    Ok(combined)
}

fn combine_samples(left: DerivedSample, right: DerivedSample, mode: CombineMode) -> DerivedSample {
    let value = match mode {
        CombineMode::Replace => right.value,
        CombineMode::Add | CombineMode::Average => left.value + right.value,
        CombineMode::Maximum => left.value.max(right.value),
        CombineMode::Minimum => left.value.min(right.value),
    };
    DerivedSample {
        value,
        confidence: left.confidence.min(right.confidence),
        state: if state_rank(left.state) >= state_rank(right.state) {
            left.state
        } else {
            right.state
        },
        sample_capture_timestamp_ns: left
            .sample_capture_timestamp_ns
            .min(right.sample_capture_timestamp_ns),
        sample_age_ns: left.sample_age_ns.max(right.sample_age_ns),
    }
}

const fn state_rank(state: SignalState) -> u8 {
    match state {
        SignalState::Observed => 0,
        SignalState::Fused => 1,
        SignalState::Predicted => 2,
        SignalState::Occluded => 3,
        SignalState::OutOfFrame => 4,
        SignalState::TrackingLost => 5,
        SignalState::Unsupported => 6,
    }
}

fn piecewise(value: f32, points: &[CurvePoint]) -> f32 {
    if value <= points[0].input {
        return points[0].output;
    }
    for pair in points.windows(2) {
        if value <= pair[1].input {
            let ratio = (value - pair[0].input) / (pair[1].input - pair[0].input);
            return pair[0].output + ratio * (pair[1].output - pair[0].output);
        }
    }
    points.last().expect("validated non-empty points").output
}
