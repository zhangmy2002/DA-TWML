$ErrorActionPreference = "Stop"

$DataRoot = if ($env:DATA_ROOT) { $env:DATA_ROOT } else { "D:\南京大学岩石教学薄片显微图像数据集" }
$MetaDir = if ($env:META_DIR) { $env:META_DIR } else { $DataRoot }

python .\src\rock_fewshot_datwml_experiments.py `
  --data_root "$DataRoot" `
  --meta_dir "$MetaDir" `
  --out_dir runs_acml `
  --experiment ablation `
  --methods protonet random_topk loss_only loss_u loss_d loss_c loss_u_d loss_u_c full `
  --ways 5 --shots 1 --query 5 `
  --seeds 0 1 2 `
  --epochs 30 --episodes_per_epoch 100 `
  --val_episodes 200 --test_episodes 1000 `
  --meta_batch 4 --topk 2 --image_size 84
