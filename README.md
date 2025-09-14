# AI Agent Evaluation Platform

A comprehensive platform for automated evaluation of AI agent responses using multi-metric analysis, built with React TypeScript frontend and Python backend with machine learning capabilities.

## Overview

The AI Agent Evaluation Platform provides automated, multi-dimensional assessment of AI-generated responses across seven key metrics: instruction following, hallucination detection, assumption analysis, coherence, accuracy, completeness, and overall quality. The platform combines traditional machine learning approaches with modern large language models to deliver transparent, explainable evaluations at scale.

## Features

- **Multi-Metric Evaluation**: Simultaneous assessment across seven distinct quality dimensions
- **Real-time Scoring**: Sub-2-second response time for individual evaluations
- **Automated Reference Generation**: AI-powered creation of ground truth answers
- **Interactive Dashboard**: Comprehensive analytics and visualization interface
- **Prompt Optimization**: Intelligent suggestions for improving low-performing prompts
- **Agent Comparison**: Head-to-head performance analysis between different AI agents
- **Export Capabilities**: Multiple format support including CSV, JSON, and PDF reports
- **Enterprise API**: RESTful API with comprehensive documentation

## Architecture

### Backend Components

- **main.py**: Core FastAPI application with routing and middleware configuration
- **Agentic_AI_Evaluator_BERT**: Fine-tuned BERT model for multi-head classification
- **trained_agentic_evaluator**: Trained model weights and configuration files

### Frontend Components

The React TypeScript frontend is organized into modular components:

- **AIAssistant.tsx**: Integrated AI chat interface for real-time assistance
- **AIEvaluationPanel.tsx**: Primary evaluation interface for prompt-response pairs
- **AnimatedCounter.tsx**: Dynamic numerical displays with smooth animations
- **ComprehensiveDashboard.tsx**: Main analytics and visualization dashboard
- **EvaluationHistory.tsx**: Historical evaluation tracking and management
- **SaveEvaluationModal.tsx**: Modal interface for saving evaluation results
- **UploadPanel.tsx**: File upload interface for batch evaluations

### Core Application

- **App.tsx**: Main application component with routing and state management
- **main.tsx**: Application entry point and React DOM rendering
- **index.css**: Global styling and design system
- **vite-env.d.ts**: TypeScript environment declarations for Vite

## Dataset

The platform includes a comprehensive evaluation dataset with:

- 240 curated prompt-response pairs
- 30 unique prompts across 10 knowledge domains
- 8 different AI agents representing various capability levels
- Multi-dimensional scoring across all quality metrics

The dataset covers diverse areas including:
- Historical facts and analysis
- Scientific concepts and explanations  
- Mathematical problem-solving
- Logical reasoning and paradoxes
- Creative writing with constraints
- Ethical and philosophical questions

## Model Architecture

The evaluation system employs a fine-tuned BERT-base model with a multi-head architecture:

- **Base Model**: BERT-base-uncased with 12 transformer layers
- **Architecture**: Custom multi-head classification heads for each metric
- **Input Processing**: Concatenated prompt, response, and reference answer
- **Output**: Seven-dimensional quality assessment with confidence scores
- **Performance**: 92% accuracy on hallucination detection, 79-100% across other metrics

## Installation

### Prerequisites

- Node.js 16+ and npm/yarn
- Python 3.8+
- CUDA-compatible GPU (recommended for model inference)

### Backend Setup

```bash
cd backend
pip install -r requirements.txt
python main.py
```

The backend server will start on `http://localhost:8000`

### Frontend Setup

```bash
npm install
npm run dev
```

The frontend development server will start on `http://localhost:5173`

### Environment Configuration

Create a `.env` file in the root directory:

```
VITE_API_BASE_URL=http://localhost:8000
OPENAI_API_KEY=your_openai_api_key_here
MODEL_PATH=./model/trained_agentic_evaluator
```

## Usage

### Web Interface

1. Navigate to the application URL in your browser
2. Use the Upload Panel to submit prompt-response pairs for evaluation
3. View real-time results in the AI Evaluation Panel
4. Access comprehensive analytics through the Dashboard
5. Export results in your preferred format

### API Integration

The platform provides a RESTful API for programmatic access:

```bash
# Single evaluation
POST /api/evaluate
{
  "prompt": "Your prompt here",
  "response": "AI response to evaluate",
  "reference": "Optional reference answer"
}

# Batch evaluation
POST /api/evaluate/batch
{
  "evaluations": [
    {
      "prompt": "Prompt 1",
      "response": "Response 1"
    }
  ]
}
```

### Python SDK Example

```python
from ai_evaluator import EvaluatorClient

client = EvaluatorClient(api_key="your_api_key")

result = client.evaluate(
    prompt="What is the capital of France?",
    response="The capital of France is Paris.",
    include_explanations=True
)

print(result.overall_score)
print(result.metric_breakdown)
```

## Model Training

To retrain the evaluation model with custom data:

```bash
cd model
python train_evaluator.py --data_path ../dataset/training_data.csv --epochs 10 --batch_size 16
```

Training hyperparameters:
- Learning rate: 2e-5 with linear decay
- Batch size: 16
- Epochs: 10 with early stopping
- Optimizer: AdamW with weight decay (0.01)
- Loss function: Weighted CrossEntropy

## Development

### Project Structure

```
├── backend/
│   ├── main.py              # FastAPI application
│   ├── requirements.txt     # Python dependencies
│   └── model/              # ML model files
├── src/
│   ├── components/         # React components
│   ├── App.tsx            # Main application
│   ├── main.tsx           # Entry point
│   └── index.css          # Global styles
├── dataset/               # Evaluation dataset
├── node_modules/          # Node.js dependencies
├── package.json          # Project configuration
├── vite.config.js        # Vite build configuration
└── README.md             # This file
```

### Code Quality

The project maintains code quality through:
- TypeScript strict mode for type safety
- ESLint configuration for code consistency
- Prettier for automated code formatting
- Comprehensive error handling and logging

### Testing

Run the test suite:

```bash
# Frontend tests
npm run test

# Backend tests
cd backend
python -m pytest tests/
```

## Performance Metrics

Current model performance benchmarks:

| Metric | Accuracy | F1 Score | Precision | Recall |
|--------|----------|----------|-----------|---------|
| Instruction Following | 83% | 88% | 85% | 91% |
| Hallucination Detection | 92% | 75% | 88% | 65% |
| Assumption Analysis | 83% | 69% | 71% | 67% |
| Coherence Assessment | 79% | 88% | 84% | 92% |
| Accuracy Verification | 79% | 85% | 82% | 88% |
| Completeness Analysis | 79% | 84% | 81% | 87% |
| Overall Quality | 100% | 100% | 100% | 100% |

## Contributing

We welcome contributions to improve the platform. Please follow these guidelines:

1. Fork the repository and create a feature branch
2. Ensure all tests pass and add tests for new functionality
3. Follow the established code style and conventions
4. Submit a pull request with a clear description of changes

### Development Guidelines

- Use TypeScript for all frontend code with proper type annotations
- Follow React best practices and hooks patterns
- Implement proper error handling and user feedback
- Write comprehensive tests for new features
- Update documentation for API changes

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Support

For technical support, feature requests, or general questions:

- Create an issue in the GitHub repository
- Contact the development team
- Check the documentation wiki for common questions

## Roadmap

Planned features and improvements:

- Multi-language evaluation support
- Advanced visualization and analytics
- Integration with popular AI platforms
- Enhanced model architectures
- Real-time collaboration features
- Mobile application development

## Citation

If you use this platform in your research, please cite:

```
@software{ai_agent_evaluator_2025,
  title={AI Agent Evaluation Platform: Multi-Metric Analysis for Automated Response Assessment},
  year={2025},
  url={https://github.com/your-username/ai-agent-evaluator}
}
```

## Acknowledgments

This project builds upon research in natural language processing, machine learning evaluation metrics, and human-computer interaction. We acknowledge the contributions of the open-source community and the researchers whose work made this platform possible.