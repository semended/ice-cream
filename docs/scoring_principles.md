# Scoring Principles

## Main idea

The score must reflect business usefulness, not only mathematical closeness.

Critical business errors should be punished harder than minor visual disagreements.

## High-priority fields

The most important fields are:

1. KIK presence
2. final outlet status
3. KIK share
4. SKU families
5. monobrand block
6. mixed with competitors
7. POSM
8. empty sections
9. foreign products

## Count fields

Absolute error is not enough.

Bad example:

field_score = 1 - min(abs(pred - gt), 10) / 10

Problem:
If gt = 2 and pred = 0, the score is 0.8, even though the model completely missed the object.

Better logic:
Use relative error when gt is small.

Suggested principle:

- if gt = 0 and pred = 0: full score;
- if gt = 0 and pred > 0: strong penalty;
- if gt > 0 and pred = 0: strong penalty;
- otherwise use relative error.

## SKU families

All SKU families should have equal weight unless there is a clear business reason.

Large and visually obvious formats may receive slightly lower reward if needed.

Example:
poleno and bucket are easier to notice, so they should not dominate the score.

## Status

Final outlet status should have high weight because it is the main business decision.

Wrong status should be penalized strongly.
