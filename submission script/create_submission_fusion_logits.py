import pandas as pd
import torch


logits1 = pd.read_csv("submissionLogits1.csv")  # columns : ID, logit_class_0 ... logit_class_6
logits2 = pd.read_csv("submissionLogits2.csv") # Paths must be adapted to your files
logits3 = pd.read_csv("submissionLogits3.csv")

merged_logits = logits1.copy()
for i in range(7):
    merged_logits[f"logit_class_{i}"] = (
        logits1[f"logit_class_{i}"] +
        logits2[f"logit_class_{i}"] +
        logits3[f"logit_class_{i}"]
    ) / 3

regression1 = pd.read_csv("submissionRegression.csv")   # Path must be adapted to your file
reg_map1 = dict(zip(regression1["ID"], regression1["views"]))

CLASS_INTERVALS = {
    0: (0, 5_000),
    1: (5_000, 10_000),
    2: (10_000, 25_000),
    3: (25_000, 50_000),
    4: (50_000, 100_000),
    5: (100_000, 500_000),
    6: (500_000, None),
}

def class_from_views(views: float) -> int:
    if views < 5_000: return 0
    if views < 10_000: return 1
    if views < 25_000: return 2
    if views < 50_000: return 3
    if views < 100_000: return 4
    if views < 500_000: return 5
    return 6

rows = []
softmax = torch.nn.Softmax(dim=0)

for _, row in merged_logits.iterrows():
    vid = row["ID"]
    logit_vector = torch.tensor([row[f"logit_class_{i}"] for i in range(7)])
    prob_vector = softmax(logit_vector)
    pred_class = int(torch.argmax(prob_vector))

    reg_view = reg_map1[vid]  
    reg_class = class_from_views(reg_view)

    if pred_class == reg_class:
        final_view = reg_view
    elif reg_class < pred_class:
        left = CLASS_INTERVALS[pred_class][0]
        final_view = prob_vector[pred_class].item() * left + prob_vector[reg_class].item() * reg_view
    else:  
        right = CLASS_INTERVALS[pred_class][1]
        if right is None:
            right = reg_view
        final_view = prob_vector[pred_class].item() * right + prob_vector[reg_class].item() * reg_view

    rows.append((vid, max(0.0, final_view)))

submission = pd.DataFrame(rows, columns=["ID", "views"])
submission.to_csv("submissionFusionLogits.csv", index=False)
