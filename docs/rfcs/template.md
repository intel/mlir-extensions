# IMEX Design Document (RFC)

## Summary
Short description of proposed solution.

## Motivation
Explain motivation for the proposed idea.
If you are proposing an optimization pass, please explain the applicable scenario and expected performance gain. It needs to address below criteria:
- The new dialect needs to introduce unique features which we don’t think upstream dialect has interests to cover.
- The new dialect should facilitate certain compile-time optimizations which otherwise cannot or very hard to achieve. In other words, there should be compile-time optimizations which requires the computation graph to be represented using the new dialects. If there is no compile-time optimization required, the functions could be represented  as a standardized MLIR module to facilitate code reuse.

## Proposal
A Full and detailed description of proposal.

## Alternative
Proposal needs to mention any alternative that were being considered with pros and cons. Instead of creating a new dialect to introduce intel specific ops, consider alternative  to add specific ops in the existing upstream dialects as extension. For optimization passes, describe alternative optimization approaches.

## Questions
Mention open questions here.
