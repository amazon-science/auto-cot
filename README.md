# Auto-CoT: Automatic Chain of Thought Prompting in Large Language Models

Cheer AI up with the "let's think step by step" prompt? More plz. *Let’s think not just step by step, but also one by one.*

Auto-CoT uses more cheers & diversity to SAVE huge manual efforts in chain of thought prompt design, matching or even exceeding performance of manual design on GPT-3.

Check out our [25-page paper](https://arxiv.org/pdf/2210.03493.pdf) for more information.

![](https://user-images.githubusercontent.com/22279212/194787183-a1f8dff8-a0ad-43a1-827f-819671503860.png)

![](https://user-images.githubusercontent.com/22279212/194787130-d28c9191-588c-41d2-a259-62377f19c934.png)


## Requirements

Python>=3.8
```
pip install torch==1.8.2+cu111 torchtext==0.9.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
pip install -r requirements.txt
```

## Datasets

Download the datasets from the following:

```
https://github.com/kojima-takeshi188/zero_shot_cot/tree/main/dataset
https://github.com/kojima-takeshi188/zero_shot_cot/tree/main/log
```

## Quick Start

See ```try_cot.ipynb```

## Instructions

Construct Demos:

```
python run_demo.py \
--task multiarith \
--pred_file log/multiarith_zero_shot_cot.log \
--demo_save_dir demos/multiarith
```

Run inference:

```
python run_inference.py \
--dataset multiarith \
--demo_path demos/multiarith \
--output_dir experiment/multiarith
```

## Citing Auto-CoT
```
@article{zhang2022automatic,
  title={Automatic Chain of Thought Prompting in Large Language Models},
  author={Zhang, Zhuosheng and Zhang, Aston and Li, Mu and Smola, Alex},
  journal={arXiv preprint arXiv:2210.03493},
  year={2022}
}
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
