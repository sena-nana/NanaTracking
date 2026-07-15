# NanaTracking fusion

`nana-tracking-fusion` combines already-normalized NTP producer results. It has no sensor SDK,
training framework, inference backend, or transport dependency. Inputs must describe the exact
same captured frame (`session_id`, `generation`, `sequence`, and capture timestamp); arrival order
is never used as synchronization.

The reference input owns stable head, eye, look-at, and normalized face geometry when available.
The extension input can fill missing scalar or structured state and can add any registered Full or
optional signal. Conflicting values are selected by signal category and confidence policy; they are
never numerically averaged. The public result contains only NTP value, confidence, and state.
