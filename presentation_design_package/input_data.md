# Ice Cream VLM MVP — Design Input Data

## Project context

We are building an MVP system for AI/VLM analysis of ice cream freezer photos.

Brand: “Коровка из Кореновки” / KIK.

The system converts field photos into business signals:
- whether KIK products are present;
- visible KIK SKU count;
- approximate KIK share among all ice cream;
- KIK SKU groups;
- POSM presence;
- monobrand block presence;
- whether KIK is mixed with competitors;
- non-ice-cream products;
- empty sections;
- final outlet status: normal / attention / critical;
- business recommendation for the field team.

Main positioning:
We are not selling “a model recognizes ice cream”.
We are showing: “the system turns thousands of field photos into manageable business signals for brand presence.”

## Audience

Business audience.

Goal:
Get budget and agree the next MVP stage.

## Current situation

No frontend interface yet.
The presentation itself should look like a product demo.

Available:
- real freezer photos;
- 10 labeled photos;
- Gemma 4 31B results;
- JSON model outputs;
- batch table;
- initial quality metrics.

Target architecture:
- segmentation model for stable share/zone/empty-section/mixing calculations;
- VLM for complex business factors: POSM, foreign products, visual context, explanation, recommendation;
- business rules for final status.

## Hero case

Photo: photo_006.jpg

Why hero:
- KIK detected;
- Gemma KIK share: 35%;
- GT KIK share: 30%;
- status matched: attention;
- KIK is mixed with competitors;
- no monobrand block;
- no POSM;
- clear business recommendation: group KIK into a block and improve presence.

Design note:
Do not emphasize SKU count because Gemma overestimated it: Gemma 12 vs GT 4.
Main emphasis: KIK detected, share is close, status matched, recommendation is useful.

## Problem case

Photo: photo_004.jpg

Why problem case:
- difficult photo: glare / partial angle;
- Gemma detected KIK but underestimated share;
- Gemma KIK share: 30%;
- GT KIK share: 100%;
- Gemma also failed on POSM, monobrand block, mixing and status.

Design note:
Use this case to show why VLM alone is not enough and why segmentation is needed.

## Current metrics

- JSON valid rate: 100%
- Schema valid rate: 100%
- KIK present F1: 1.0
- KIK share MAE: 16.5 percentage points
- SKU count MAE: 4.1
- Status is often underestimated as normal

## Status mapping

- 0 = normal
- 1 = attention
- 2 = critical
