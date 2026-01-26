> [!NOTE]
> This directory is largely dependent on the "model-organisms-for-EM" repository by Turner and Solingo et al. (2025).<br>
> **Original Repo**: https://github.com/clarifying-EM/model-organisms-for-EM/tree/main
> **Paper by Turner et al. (2025)**: https://arxiv.org/abs/2506.11613
> **Paper by Solingo et al. (2025)**: https://arxiv.org/abs/2506.11618

## Overview
This directory contains the evaluation scripts that queries an OpenAI model (GPT4o) to judge the alignment and other metrics of the EM models' responses.

## Run
```bash
uv run gen_judge_responses.py 
```