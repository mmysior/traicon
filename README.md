# TRAICON

TRAICON (TRIZ AI CONtradiction solver) is an AI-powered framework that helps solve technical contradictions using TRIZ methodology. It leverages various LLM providers (OpenAI, Groq, Ollama) to analyze problems and generate innovative solutions based on TRIZ principles.

## Features

- 🤖 Multiple LLM provider support (OpenAI, Groq, Ollama)
- 🧮 Automated technical contradiction analysis
- 💡 TRIZ-based solution generation
- 🎯 Parameter mapping and principle matching
- 🌐 Interactive Streamlit interface

## Installation

1. Clone the repository:

```
git clone https://github.com/mmysior/traicon.git
cd traicon
```

2. Create and activate a Conda environment:

```
make create_environment
conda activate traicon
```

3. Install the required dependencies:

```
make create_environment
```

or alternatively with conda:

```
conda env create --name traicon -f environment.yml
```

4. Create a `.env` file based on the example in the project root and add your API keys:

```
OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key
```

## Usage

You can start the application in one of two ways:

1. Using Make:

```
make run
```

2. Using Streamlit directly:

```
streamlit run app/main.py
```

The app will be available at `http://localhost:8501`

## How It Works

TRAICON helps solve technical contradictions through the following process:

1. **Problem Analysis**: Enter your technical problem description
2. **Contradiction Extraction**: The system identifies the core technical contradiction
3. **Parameter Mapping**: Maps the contradiction to standard TRIZ parameters
4. **Solution Generation**: Generates solutions based on relevant TRIZ principles
5. **Result Presentation**: Displays structured solutions with explanations

## Observability with Langfuse

TRAICON supports tracing and monitoring of LLM interactions using Langfuse. To enable tracing:

1. Sign up for a free account at [Langfuse](https://langfuse.com)

2. Add these environment variables to your `.env` file:

```
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com
```

With Langfuse enabled, you can:
- Monitor all LLM interactions
- Track token usage and costs
- Analyze response times and quality
- Debug model behaviors
- Generate usage analytics

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

