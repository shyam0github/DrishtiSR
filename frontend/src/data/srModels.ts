import { FACTS } from "./facts";

/**
 * Parameter counts of well-known super-resolution networks, for the "how big
 * is typical" comparison. The one list for the whole site: /novelty/efficiency
 * reads it now and /compare must read it too, so the two pages cannot disagree.
 *
 * `literature` counts are as reported for the ×4 RGB models by their papers
 * (or the standard comparison tables that quote them). They were not measured
 * in this repo. Our model takes 4 bands (B04 B03 B02 B08), so its first and
 * last convolutions are slightly larger than an RGB model's would be.
 */
export type SrModelSource = "ours" | "in-repo" | "literature";

export interface SrModelRef {
  name: string;
  params: number;
  source: SrModelSource;
  /** Paper, or the repo file the number comes from. */
  cite: string;
}

export const SR_MODELS: readonly SrModelRef[] = [
  { name: "DrishtiSR (A2)", params: FACTS.params, source: "ours", cite: "reports/mvp/deploy_table.md" },
  { name: "EDSR-baseline (Run A)", params: FACTS.runAParams, source: "in-repo", cite: "reports/day2_runA.md" },
  { name: "SRCNN", params: 57_000, source: "literature", cite: "Dong et al., ECCV 2014" },
  { name: "VDSR", params: 665_000, source: "literature", cite: "Kim et al., CVPR 2016" },
  { name: "IMDN", params: 715_000, source: "literature", cite: "Hui et al., ACM MM 2019" },
  { name: "SwinIR-light", params: 897_000, source: "literature", cite: "Liang et al., ICCVW 2021" },
  { name: "CARN", params: 1_592_000, source: "literature", cite: "Ahn et al., ECCV 2018" },
  { name: "SwinIR", params: 11_900_000, source: "literature", cite: "Liang et al., ICCVW 2021" },
  { name: "RCAN", params: 15_600_000, source: "literature", cite: "Zhang et al., ECCV 2018" },
  { name: "ESRGAN (RRDBNet)", params: 16_700_000, source: "literature", cite: "Wang et al., ECCVW 2018" },
  { name: "EDSR", params: 43_000_000, source: "literature", cite: "Lim et al., CVPRW 2017" },
];
