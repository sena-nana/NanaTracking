#![no_main]

use libfuzzer_sys::fuzz_target;
use nana_tracking_protocol::{CanonicalCodec, NanaTrackingDescriptor, NanaTrackingResult};

fuzz_target!(|data: &[u8]| {
    let _ = CanonicalCodec::decode::<NanaTrackingDescriptor>(data);
    let _ = CanonicalCodec::decode::<NanaTrackingResult>(data);
});

