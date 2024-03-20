# PrivacyOracle

## Introduction


## Setup

In order to query ChatGPT, you need to acquire an [OpenAI API key](https://platform.openai.com/api-keys).  Save the secret key to a document called "API_KEY" without any file extension.

To run the private state experiments, you also need to download the [ADL dataset](https://archive.ics.uci.edu/dataset/271/activities+of+daily+living+adls+recognition+using+binary+sensors) from the UCI ML repo.  Please move this to the *datasets* folder.

Once you have completed these steps, you should see the following folder and file structure:

```
PrivacyOracle/
├── API_KEY
├── datasets/
│   └── UCI ADL Binary Dataset
├── prompts
├── query
└── run_queries.py
```
