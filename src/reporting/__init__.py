"""P7 reporting: post-hoc checkpoint selection and the Day 6 tables.

``selection`` wraps the pre-registered Day 3 rule
(:func:`src.eval.ckpt_selection.select_checkpoint`, config block
``eval_all_ckpts.selection``) and writes one auditable record per run.
``tables`` assembles the ablation / deployment / uncertainty tables and the
headline numbers from evidence files that already exist; it never evaluates a
model itself.
"""
