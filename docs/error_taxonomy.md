# VLM Error Taxonomy

## 1. False positive

Model sees KIK when KIK is not actually visible.

## 2. False negative

Model misses KIK when KIK is visible.

## 3. SKU count error

Model gives wrong number of visible KIK SKUs.

## 4. SKU-family error

Model misclassifies product families.

Example:
The model confuses bricket and poleno.

## 5. KIK share error

Model estimates KIK share incorrectly.

Important:
KIK share should mean:

KIK / (KIK + competitors)

not:

KIK / entire freezer volume.

## 6. Perspective/photo-angle issue

The error is likely caused by bad angle, strong perspective, or distant products.

## 7. Small-object visibility issue

The model misses small or rare objects.

Example:
Only one KIK bucket is visible.

## 8. Business-status interpretation error

The model detects visual facts correctly but assigns the wrong final status.

## 9. JSON/schema error

The model output is invalid JSON or does not match the expected schema.

## 10. Hallucination

The model invents visible products, SKU groups, POSM, or other facts.
