# Model provenance

`ant.xml` is copied byte-for-byte from the Gymnasium 1.3.0 package's
`gymnasium/envs/mujoco/assets/ant.xml` (SHA-256
`cd5f83ef0ea35b0969e65d360c5bacd5b74ccaef6b27e4433b5168c605e3e2be`).
Gymnasium is distributed under the MIT license; see its
[repository and notices](https://github.com/Farama-Foundation/Gymnasium/tree/v1.3.0).

The Humanoid experiment deliberately uses
`benchmarks/humanoid/humanoid.xml` from the checked-out MJWarp PR #1535 head,
SHA-256 `9b29ea10152748e66a0a77f7651efbbb8282fef9621e7cb8d220ca157c41e61a`.
It is not duplicated here, which makes the experiment fail closed if the
expected PR checkout is missing.
