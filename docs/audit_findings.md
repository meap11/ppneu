# Audit findings (Step 1, official Kaggle copy, 5,856 images)

- **Exact duplicates:** 30 sets, 32 redundant images (5,824 unique). All within one
  split, same label, same filename group. None cross train/test.
- **Filename IDs, fine (namespaces kept separate):** 0 groups shared between train and test.
- **Filename IDs, coarse (bacteria/virus merged, IM/NORMAL2-IM merged):** 264 groups,
  416 test images shared with train. Every one is a cross-namespace collision
  (test virus vs train bacteria: 107; test NORMAL2-IM vs train IM: 94;
  test bacteria vs train virus: 63). So the "264 overlapping patients" figure
  circulating online is produced by merging namespaces, and filenames cannot tell
  us whether those are the same children. Inside train, 808 person IDs carry both
  bacteria and virus images, so a shared numbering is plausible; the test folder
  numbering looks independent (test bacteria IDs only 78-175).
- **Perceptual hashing cannot verify patient identity on these CXRs:** same-ID pairs
  (median 22 bits) look almost like random pairs (median 24). Pairs at distance <= 2
  were visibly different children. Near-duplicate counts are therefore NOT reported
  as findings.
- **Unparsed names (144):** extra numeric suffixes (`IM-0001-0001-0001`,
  `person1_virus_6_2`); parser fixed, 0 unparsed now.

## Consequence for the paper
The official split shows no verifiable patient leakage. The measurable leakage is in
the common practice of pooling folders and re-splitting at the image level:
605 of 879 test images (69%) then share a conservative patient cluster with train
(`manifests/leakage_report.json`). The experiment compares official vs
image_random vs grouped.
