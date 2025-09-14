from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import torch
import traceback
from transformers import BertTokenizer
from datetime import datetime
import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io
import base64
import tempfile
import sqlite3
import uuid
from contextlib import contextmanager
from openai import OpenAI
from enum import Enum

# Enhanced logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Response Evaluation API with AI Assistant", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
OPENAI_CHEAP_MODEL = os.getenv("OPENAI_CHEAP_MODEL", "gpt-3.5-turbo")

# # Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)  

# Database setup
DATABASE_PATH = "evaluation_history.db"

def init_database():
    """Initialize SQLite database for storing evaluation history and AI assistant data"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Create evaluations table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS evaluations (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            total_evaluated INTEGER,
            total_agents INTEGER,
            avg_overall_score REAL,
            data_json TEXT,
            metadata_json TEXT,
            ai_generated BOOLEAN DEFAULT FALSE
        )
    ''')
    
    # Create evaluation_results table for detailed results
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS evaluation_results (
            id TEXT PRIMARY KEY,
            evaluation_id TEXT,
            prompt_id TEXT,
            agent_id TEXT,
            prompt TEXT,
            response TEXT,
            reference TEXT,
            instruction_score REAL,
            hallucination_score REAL,
            assumption_score REAL,
            coherence_score REAL,
            accuracy_score REAL,
            completeness_score REAL,
            overall_score REAL,
            generated_at TIMESTAMP,
            ai_generated_reference BOOLEAN DEFAULT FALSE,
            ai_explanation TEXT,
            suggested_prompt TEXT,
            FOREIGN KEY (evaluation_id) REFERENCES evaluations (id)
        )
    ''')
    
    # Create AI assistant conversations table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ai_conversations (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            user_message TEXT,
            ai_response TEXT,
            context_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            evaluation_id TEXT,
            FOREIGN KEY (evaluation_id) REFERENCES evaluations (id)
        )
    ''')
    
    # Create AI assistant feedback table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ai_feedback (
            id TEXT PRIMARY KEY,
            result_id TEXT,
            metric_name TEXT,
            explanation TEXT,
            improvement_suggestions TEXT,
            suggested_prompt TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (result_id) REFERENCES evaluation_results (id)
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("✅ Database initialized successfully")

@contextmanager
def get_db():
    """Database connection context manager"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row  # Enable dict-like access
    try:
        yield conn
    finally:
        conn.close()

# Initialize database on startup
init_database()

# Model definition (keeping existing model for compatibility)
import torch.nn as nn
from transformers import BertModel

class MultiHeadBERT(nn.Module):
    def __init__(self, out_dim=7):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.head = nn.Linear(self.bert.config.hidden_size, out_dim)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.pooler_output
        return self.head(cls)

# Global variables
tokenizer = None
model = None

# Enhanced Pydantic Models
class PromptItem(BaseModel):
    prompt_id: str
    prompt: str
    agent_id: str
    response: str
    reference: Optional[str] = None  # Now optional - can be generated by AI

class ExportRequest(BaseModel):
    data: Dict[str, Any]
    export_type: str
    include_charts: bool = True
    timestamp: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class ExplanationRequest(BaseModel):
    result: Dict[str, Any]
    metric: str
    context: Optional[Dict[str, Any]] = None

class SaveEvaluationRequest(BaseModel):
    data: Dict[str, Any]
    name: str
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class AIAssistantRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    evaluation_id: Optional[str] = None

class AIEvaluationRequest(BaseModel):
    prompt: str
    response: str
    agent_id: str
    reference: Optional[str] = None
    generate_reference: bool = True
    include_explanations: bool = True
    include_suggestions: bool = True

class BulkAIEvaluationRequest(BaseModel):
    items: List[AIEvaluationRequest]
    evaluation_name: Optional[str] = None
    evaluation_description: Optional[str] = None

class PromptImprovementRequest(BaseModel):
    prompt: str
    response: str
    reference: str
    scores: Dict[str, float]
    target_metric: Optional[str] = None

class EvaluationMode(str, Enum):
    TRADITIONAL = "traditional"
    AI_ASSISTED = "ai_assisted"
    HYBRID = "hybrid"

@app.on_event("startup")
async def startup_event():
    global tokenizer, model
    try:
        logger.info("=== Starting Enhanced Model Loading ===")
        
        # Check OpenAI API key
        if not openai_client:
            logger.warning("⚠️ OpenAI API key not configured. AI features will be limited.")
        else:
            logger.info("✅ OpenAI client initialized")
        
        # Load traditional model (optional now)
        model_path = "trained_agentic_evaluator.pt"
        if os.path.exists(model_path):
            logger.info(f"✅ Traditional model file found: {model_path}")
            
            # Load tokenizer
            logger.info("🔄 Loading BERT tokenizer...")
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            logger.info("✅ Tokenizer loaded successfully")
            
            # Load model
            logger.info("🔄 Loading traditional evaluation model...")
            model = MultiHeadBERT(out_dim=7)
            
            # Load state dict with error handling
            state_dict = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state_dict)
            model.eval()
            
            logger.info("✅ Traditional model loaded successfully")
        else:
            logger.info("ℹ️ Traditional model not found. Using AI-only evaluation.")
        
        logger.info("=== Startup Complete ===")
        
        # Set matplotlib backend for headless environments
        plt.switch_backend('Agg')
        
    except Exception as e:
        logger.error(f"❌ Startup error: {str(e)}")
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        raise

# AI Assistant Functions
async def call_openai_api(messages: List[Dict[str, str]], model: str = None, temperature: float = 0.7) -> str:
    """Call OpenAI API with error handling"""
    if not openai_client:
        raise HTTPException(status_code=500, detail="OpenAI API not configured")
    
    try:
        response = openai_client.chat.completions.create(
            model=model or OPENAI_CHEAP_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI service error: {str(e)}")

async def generate_reference_answer(prompt: str, context: str = None) -> str:
    """Generate reference answer using LLM"""
    messages = [
        {
            "role": "system",
            "content": "You are an expert evaluator. Generate a comprehensive, accurate reference answer for the given prompt. The reference should be well-structured, complete, and serve as a gold standard for evaluation."
        },
        {
            "role": "user",
            "content": f"Prompt: {prompt}\n\n{f'Context: {context}' if context else ''}Generate a high-quality reference answer:"
        }
    ]
    
    return await call_openai_api(messages, model=OPENAI_MODEL, temperature=0.3)

async def generate_ai_scores(prompt: str, response: str, reference: str) -> Dict[str, float]:
    """Generate scores using LLM as judge"""
    scoring_prompt = f"""
    You are an expert AI evaluator. Score the following response against the reference on these 7 metrics (scale 0.0 to 1.0):

    1. instruction_score: How well does the response follow the given instructions? ( lower is worse, higher is better)
    2. hallucination_score: How accurate and free from false information is the response? (lower is better, higher is worse)
    3. assumption_score: How appropriately does the response handle assumptions? ( lower is worse, higher is better)
    4. coherence_score: How logical, clear, and well-structured is the response? ( lower is worse, higher is better)
    5. accuracy_score: How factually correct and precise is the response?  ( lower is worse, higher is better)
    6. completeness_score: How thoroughly does the response address all aspects? ( lower is worse, higher is better)
    7. overall_score: Overall quality combining all aspects above. ( lower is worse, higher is better)

    PROMPT: {prompt}

    REFERENCE ANSWER: {reference}

    RESPONSE TO EVALUATE: {response}

    Provide scores in this exact JSON format:
    {{
        "instruction_score": 0.0,
        "hallucination_score": 0.0,
        "assumption_score": 0.0,
        "coherence_score": 0.0,
        "accuracy_score": 0.0,
        "completeness_score": 0.0,
        "overall_score": 0.0
    }}
    """
    
    messages = [
        {"role": "system", "content": "You are a precise AI evaluator. Respond only with valid JSON."},
        {"role": "user", "content": scoring_prompt}
    ]
    
    response_text = await call_openai_api(messages, model=OPENAI_MODEL, temperature=0.1)
    
    try:
        # Clean response and parse JSON
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:-3]
        elif response_text.startswith("```"):
            response_text = response_text[3:-3]
        
        scores = json.loads(response_text)
        
        # Validate scores are in range [0, 1]
        for key, value in scores.items():
            scores[key] = max(0.0, min(1.0, float(value)))
        
        return scores
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Error parsing AI scores: {e}")
        # Return default scores if parsing fails
        return {
            "instruction_score": 0.5,
            "hallucination_score": 0.5,
            "assumption_score": 0.5,
            "coherence_score": 0.5,
            "accuracy_score": 0.5,
            "completeness_score": 0.5,
            "overall_score": 0.5
        }

async def generate_detailed_explanation(prompt: str, response: str, reference: str, scores: Dict[str, float]) -> str:
    """Generate detailed explanation for scores"""
    explanation_prompt = f"""
    Provide a detailed explanation for why these scores were given:

    PROMPT: {prompt}
    RESPONSE: {response}
    REFERENCE: {reference}

    SCORES:
    {json.dumps(scores, indent=2)}

    Explain:
    1. Strengths of the response
    2. Areas for improvement
    3. Specific examples from the response
    4. How each metric was evaluated
    5. Overall assessment

    Be specific and actionable in your feedback.
    """
    
    messages = [
        {
            "role": "system",
            "content": "You are an expert evaluator providing detailed, constructive feedback on AI responses."
        },
        {"role": "user", "content": explanation_prompt}
    ]
    
    return await call_openai_api(messages, model=OPENAI_CHEAP_MODEL, temperature=0.5)
def create_visualizations(data: Dict[str, Any]) -> Dict[str, str]:
    """Create visualization charts and return as base64 encoded images"""
    charts = {}
    
    try:
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Agent Performance Radar Chart
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        
        agents = list(data['agent_summaries'].keys())
        metrics = ['instruction_score', 'hallucination_score', 'assumption_score', 
                  'coherence_score', 'accuracy_score', 'completeness_score']
        
        angles = [n / len(metrics) * 2 * 3.14159 for n in range(len(metrics))]
        angles += angles[:1]  # Complete the circle
        
        colors_list = ['#8884d8', '#82ca9d', '#ffc658', '#ff7c7c', '#8dd1e1']
        
        for i, agent in enumerate(agents[:5]):  # Limit to 5 agents for clarity
            values = [data['agent_summaries'][agent][metric] for metric in metrics]
            values += values[:1]  # Complete the circle
            
            color = colors_list[i % len(colors_list)]
            ax.plot(angles, values, 'o-', linewidth=2, label=agent, color=color)
            ax.fill(angles, values, alpha=0.25, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        ax.set_title('Agent Performance Comparison', size=16, fontweight='bold')
        
        # Save as base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        charts['radar'] = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        logger.info(f"✅ Created {len(charts)} visualization charts")
        return charts
        
    except Exception as e:
        logger.error(f"❌ Error creating visualizations: {str(e)}")
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        return {}
    
async def generate_prompt_improvement_suggestion(prompt: str, response: str, reference: str, scores: Dict[str, float], target_metric: str = None) -> Dict[str, str]:
    """Generate suggestions for improving the prompt"""
    low_metrics = [k for k, v in scores.items() if v < 0.7]
    focus_metric = target_metric or (low_metrics[0] if low_metrics else "overall_score")
    
    improvement_prompt = f"""
    Analyze this evaluation and suggest how to improve the prompt to get better results:

    CURRENT PROMPT: {prompt}
    RESPONSE RECEIVED: {response}
    REFERENCE ANSWER: {reference}

    SCORES: {json.dumps(scores, indent=2)}

    FOCUS ON: {focus_metric} (current score: {scores.get(focus_metric, 0.5):.2f})

    Provide:
    1. Analysis of why the current prompt led to the low score
    2. Specific prompt improvement suggestions
    3. A rewritten version of the prompt
    4. Expected improvements

    Format your response as JSON:
    {{
        "analysis": "Why the current prompt is lacking...",
        "suggestions": ["Suggestion 1", "Suggestion 2", ...],
        "improved_prompt": "Rewritten prompt here...",
        "expected_improvements": "What should improve..."
    }}
    """
    
    messages = [
        {
            "role": "system",
            "content": "You are a prompt engineering expert. Help improve prompts for better AI responses."
        },
        {"role": "user", "content": improvement_prompt}
    ]
    
    response_text = await call_openai_api(messages, model=OPENAI_MODEL, temperature=0.6)
    
    try:
        # Clean and parse JSON response
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:-3]
        elif response_text.startswith("```"):
            response_text = response_text[3:-3]
        
        return json.loads(response_text)
    except json.JSONDecodeError:
        return {
            "analysis": "Unable to generate detailed analysis",
            "suggestions": ["Consider being more specific in your instructions", "Add examples or constraints"],
            "improved_prompt": prompt,
            "expected_improvements": "Minor improvements expected"
        }

# Traditional scoring function (keeping for hybrid mode)
def predict_metrics(prompt: str, response: str, reference: str) -> List[float]:
    """Predict metrics with enhanced error handling (traditional model)"""
    if not model or not tokenizer:
        logger.warning("Traditional model not available, using default scores")
        return [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    
    try:
        with torch.no_grad():
            text_input = f"Prompt: {prompt} [SEP] Response: {response} [SEP] Reference: {reference}"
            
            inputs = tokenizer(
                text_input,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding="max_length"
            )
            
            outputs = model(inputs['input_ids'], inputs['attention_mask'])
            scores = outputs.squeeze().tolist()
            
            if isinstance(scores, float):
                scores = [scores]
            
            if len(scores) != 7:
                scores = (scores + [0.5] * 7)[:7]
            
            scores[2]=1.0 - scores[2]  # Invert assumption_score
            scores[6]=scores[6]/5.0
            scores = [max(0, min(1, float(score))) for score in scores]
            return scores
            
    except Exception as e:
        logger.error(f"❌ Traditional prediction error: {str(e)}")
        return [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

# API Endpoints

@app.get("/health")
async def health_check():
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM evaluations")
        total_evaluations = cursor.fetchone()[0]
    
    return {
        "status": "healthy",
        "message": "Enhanced API is running successfully",
        "traditional_model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "ai_assistant_available": openai_client is not None,
        "version": "3.0.0",
        "stored_evaluations": total_evaluations,
        "features": [
            "AI-powered scoring",
            "Reference generation",
            "Detailed explanations",
            "Prompt improvement",
            "Assistant chat"
        ]
    }

# NEW AI ASSISTANT ENDPOINTS

@app.post("/api/ai-assistant/chat")
async def ai_assistant_chat(request: AIAssistantRequest):
    """AI Assistant chat endpoint"""
    try:
        logger.info(f"🤖 AI Assistant chat: {request.message[:50]}...")
        
        # Build context-aware system prompt
        system_prompt = """You are an AI Assistant for an AI Agent Evaluation Dashboard. You help users:

        1. Understand evaluation metrics and scores
        2. Interpret results and comparisons
        3. Navigate the dashboard features
        4. Troubleshoot issues
        5. Provide guidance on improving AI agent performance

        Be helpful, concise, and specific. If users ask about specific evaluations or scores, use the provided context data.
        """
        
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add context if provided
        if request.context:
            context_info = f"Current dashboard context: {json.dumps(request.context, indent=2)}"
            messages.append({"role": "system", "content": context_info})
        
        messages.append({"role": "user", "content": request.message})
        
        ai_response = await call_openai_api(messages, model=OPENAI_CHEAP_MODEL, temperature=0.7)
        
        # Store conversation
        conversation_id = str(uuid.uuid4())
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO ai_conversations (id, session_id, user_message, ai_response, context_data, evaluation_id)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                conversation_id,
                request.session_id or str(uuid.uuid4()),
                request.message,
                ai_response,
                json.dumps(request.context) if request.context else None,
                request.evaluation_id
            ))
            conn.commit()
        
        return {
            "success": True,
            "response": ai_response,
            "conversation_id": conversation_id,
            "session_id": request.session_id
        }
        
    except Exception as e:
        logger.error(f"❌ AI Assistant chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Assistant error: {str(e)}")

@app.post("/api/ai-evaluation/single")
async def ai_evaluation_single(request: AIEvaluationRequest):
    """Single evaluation using AI"""
    try:
        logger.info(f"🔄 AI Evaluation: {request.prompt[:50]}...")
        
        # Generate reference if not provided
        reference = request.reference
        if not reference and request.generate_reference:
            logger.info("🔄 Generating reference answer...")
            reference = await generate_reference_answer(request.prompt)
        
        if not reference:
            raise HTTPException(status_code=400, detail="Reference answer is required")
        
        # Generate AI scores
        logger.info("🔄 Generating AI scores...")
        scores = await generate_ai_scores(request.prompt, request.response, reference)
        
        # Generate explanation if requested
        explanation = None
        if request.include_explanations:
            logger.info("🔄 Generating detailed explanation...")
            explanation = await generate_detailed_explanation(
                request.prompt, request.response, reference, scores
            )
        
        # Generate prompt improvements if requested
        suggestions = None
        if request.include_suggestions:
            logger.info("🔄 Generating prompt improvement suggestions...")
            suggestions = await generate_prompt_improvement_suggestion(
                request.prompt, request.response, reference, scores
            )
        
        result = {
            "prompt_id": f"ai_eval_{uuid.uuid4().hex[:8]}",
            "agent_id": request.agent_id,
            "prompt": request.prompt,
            "response": request.response,
            "reference": reference,
            "ai_generated_reference": not request.reference,
            "metrics": scores,
            "explanation": explanation,
            "prompt_suggestions": suggestions,
            "generated_at": datetime.now().isoformat(),
            "evaluation_mode": "ai_assisted"
        }
        
        logger.info("✅ AI evaluation completed successfully")
        return {
            "success": True,
            "result": result,
            "message": "AI evaluation completed successfully"
        }
        
    except Exception as e:
        logger.error(f"❌ AI evaluation error: {str(e)}")
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"AI evaluation failed: {str(e)}")

@app.post("/api/ai-evaluation/bulk")
async def ai_evaluation_bulk(request: BulkAIEvaluationRequest):
    """Bulk evaluation using AI"""
    try:
        logger.info(f"🔄 Bulk AI Evaluation: {len(request.items)} items")
        
        results = []
        failed_items = []
        
        for i, item in enumerate(request.items):
            try:
                logger.info(f"🔄 Processing item {i+1}/{len(request.items)}")
                
                # Generate reference if needed
                reference = item.reference
                if not reference and item.generate_reference:
                    reference = await generate_reference_answer(item.prompt)
                
                if not reference:
                    failed_items.append({"index": i, "error": "No reference available"})
                    continue
                
                # Generate scores
                scores = await generate_ai_scores(item.prompt, item.response, reference)
                
                # Generate explanation if requested
                explanation = None
                if item.include_explanations:
                    explanation = await generate_detailed_explanation(
                        item.prompt, item.response, reference, scores
                    )
                
                # Generate suggestions if requested
                suggestions = None
                if item.include_suggestions:
                    suggestions = await generate_prompt_improvement_suggestion(
                        item.prompt, item.response, reference, scores
                    )
                
                result = {
                    "prompt_id": f"bulk_ai_{uuid.uuid4().hex[:8]}",
                    "agent_id": item.agent_id,
                    "prompt": item.prompt,
                    "response": item.response,
                    "reference": reference,
                    "ai_generated_reference": not item.reference,
                    "metrics": scores,
                    "explanation": explanation,
                    "prompt_suggestions": suggestions,
                    "generated_at": datetime.now().isoformat(),
                    "evaluation_mode": "ai_assisted"
                }
                
                results.append(result)
                
            except Exception as item_error:
                logger.error(f"❌ Error processing item {i+1}: {str(item_error)}")
                failed_items.append({"index": i, "error": str(item_error)})
                continue
        
        if not results:
            raise HTTPException(status_code=500, detail="No items could be processed successfully")
        
        # Calculate summaries
        logger.info("🔄 Calculating summaries...")
        agents = {}
        for result in results:
            agent_id = result["agent_id"]
            if agent_id not in agents:
                agents[agent_id] = []
            agents[agent_id].append(result["metrics"])
        
        agent_summaries = {}
        for agent_id, metrics_list in agents.items():
            agent_summaries[agent_id] = {
                metric: round(sum(m[metric] for m in metrics_list) / len(metrics_list), 3)
                for metric in ["instruction_score", "hallucination_score", "assumption_score", 
                             "coherence_score", "accuracy_score", "completeness_score", "overall_score"]
            }
        
        response_data = {
            "success": True,
            "total_evaluated": len(results),
            "results": results,
            "agent_summaries": agent_summaries,
            "overall_summary": {
                "avg_overall_score": round(sum(r["metrics"]["overall_score"] for r in results) / len(results), 3),
                "total_agents": len(agents),
                "total_prompts": len(results)
            },
            "failed_items": failed_items,
            "evaluation_mode": "ai_assisted"
        }
        
        # Auto-save if evaluation name provided
        if request.evaluation_name:
            evaluation_id = str(uuid.uuid4())
            
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO evaluations 
                    (id, name, description, total_evaluated, total_agents, avg_overall_score, data_json, metadata_json, ai_generated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    evaluation_id,
                    request.evaluation_name,
                    request.evaluation_description,
                    len(results),
                    len(agents),
                    response_data["overall_summary"]["avg_overall_score"],
                    json.dumps(response_data),
                    json.dumps({"ai_assisted": True, "version": "3.0.0"}),
                    True
                ))
                
                # Save individual results
                for result in results:
                    result_id = str(uuid.uuid4())
                    cursor.execute('''
                        INSERT INTO evaluation_results 
                        (id, evaluation_id, prompt_id, agent_id, prompt, response, reference,
                         instruction_score, hallucination_score, assumption_score, coherence_score,
                         accuracy_score, completeness_score, overall_score, generated_at, 
                         ai_generated_reference, ai_explanation, suggested_prompt)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        result_id, evaluation_id, result['prompt_id'], result['agent_id'],
                        result['prompt'], result['response'], result['reference'],
                        result['metrics']['instruction_score'],
                        result['metrics']['hallucination_score'],
                        result['metrics']['assumption_score'],
                        result['metrics']['coherence_score'],
                        result['metrics']['accuracy_score'],
                        result['metrics']['completeness_score'],
                        result['metrics']['overall_score'],
                        result['generated_at'],
                        result.get('ai_generated_reference', False),
                        result.get('explanation'),
                        json.dumps(result.get('prompt_suggestions')) if result.get('prompt_suggestions') else None
                    ))
                
                conn.commit()
            
            response_data["evaluation_id"] = evaluation_id
        
        logger.info(f"✅ Successfully processed {len(results)} items")
        return response_data
        
    except Exception as e:
        logger.error(f"❌ Bulk AI evaluation error: {str(e)}")
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Bulk evaluation failed: {str(e)}")

@app.post("/api/prompt-improvement")
async def generate_prompt_improvement(request: PromptImprovementRequest):
    """Generate prompt improvement suggestions"""
    try:
        logger.info("🔄 Generating prompt improvement suggestions...")
        
        suggestions = await generate_prompt_improvement_suggestion(
            request.prompt, 
            request.response, 
            request.reference, 
            request.scores,
            request.target_metric
        )
        
        return {
            "success": True,
            "suggestions": suggestions,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Prompt improvement error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate improvements: {str(e)}")

@app.post("/api/generate-reference")
async def generate_reference(request: Dict[str, str]):
    """Generate reference answer for a prompt"""
    try:
        prompt = request.get("prompt")
        context = request.get("context")
        
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")
        
        logger.info(f"🔄 Generating reference for: {prompt[:50]}...")
        
        reference = await generate_reference_answer(prompt, context)
        
        return {
            "success": True,
            "reference": reference,
            "generated_at": datetime.now().isoformat(),
            "ai_generated": True
        }
        
    except Exception as e:
        logger.error(f"❌ Reference generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate reference: {str(e)}")


@app.post("/api/evaluations/save")
async def save_evaluation(request: SaveEvaluationRequest):
    """Save evaluation results to history"""
    try:
        logger.info(f"🔄 Saving evaluation: {request.name}")
        
        evaluation_id = str(uuid.uuid4())
        data = request.data
        
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Save main evaluation record
            cursor.execute('''
                INSERT INTO evaluations 
                (id, name, description, total_evaluated, total_agents, avg_overall_score, data_json, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                evaluation_id,
                request.name,
                request.description,
                data['total_evaluated'],
                data['overall_summary']['total_agents'],
                data['overall_summary']['avg_overall_score'],
                json.dumps(data),
                json.dumps(request.metadata) if request.metadata else None
            ))
            
            # Save individual results
            for result in data['results']:
                result_id = str(uuid.uuid4())
                cursor.execute('''
                    INSERT INTO evaluation_results 
                    (id, evaluation_id, prompt_id, agent_id, prompt, response, reference,
                     instruction_score, hallucination_score, assumption_score, coherence_score,
                     accuracy_score, completeness_score, overall_score, generated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    result_id, evaluation_id, result['prompt_id'], result['agent_id'],
                    result['prompt'], result['response'], result['reference'],
                    result['metrics']['instruction_score'],
                    result['metrics']['hallucination_score'],
                    result['metrics']['assumption_score'],
                    result['metrics']['coherence_score'],
                    result['metrics']['accuracy_score'],
                    result['metrics']['completeness_score'],
                    result['metrics']['overall_score'],
                    result['generated_at']
                ))
            
            conn.commit()
        
        logger.info(f"✅ Successfully saved evaluation: {evaluation_id}")
        return {
            "success": True,
            "evaluation_id": evaluation_id,
            "message": f"Evaluation '{request.name}' saved successfully"
        }
        
    except Exception as e:
        logger.error(f"❌ Save evaluation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save evaluation: {str(e)}")

@app.get("/api/evaluations")
async def get_evaluations(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """Get list of saved evaluations"""
    try:
        logger.info(f"🔄 Fetching evaluations (limit: {limit}, offset: {offset})")
        
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Get total count
            cursor.execute("SELECT COUNT(*) FROM evaluations")
            total_count = cursor.fetchone()[0]
            
            # Get evaluations with pagination
            cursor.execute('''
                SELECT id, name, description, created_at, total_evaluated, 
                       total_agents, avg_overall_score
                FROM evaluations 
                ORDER BY created_at DESC 
                LIMIT ? OFFSET ?
            ''', (limit, offset))
            
            evaluations = []
            for row in cursor.fetchall():
                evaluations.append({
                    "id": row[0],
                    "name": row[1],
                    "description": row[2],
                    "created_at": row[3],
                    "total_evaluated": row[4],
                    "total_agents": row[5],
                    "avg_overall_score": row[6]
                })
        
        logger.info(f"✅ Retrieved {len(evaluations)} evaluations")
        return {
            "success": True,
            "evaluations": evaluations,
            "pagination": {
                "total": total_count,
                "limit": limit,
                "offset": offset,
                "has_next": offset + limit < total_count
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Get evaluations error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve evaluations: {str(e)}")

@app.get("/api/evaluations/{evaluation_id}")
async def get_evaluation(evaluation_id: str):
    """Get specific evaluation by ID"""
    try:
        logger.info(f"🔄 Fetching evaluation: {evaluation_id}")
        
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Get evaluation data
            cursor.execute('''
                SELECT id, name, description, created_at, total_evaluated, 
                       total_agents, avg_overall_score, data_json, metadata_json
                FROM evaluations 
                WHERE id = ?
            ''', (evaluation_id,))
            
            row = cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Evaluation not found")
            
            # Parse the stored JSON data
            evaluation_data = json.loads(row[7])
            metadata = json.loads(row[8]) if row[8] else None
            
            result = {
                "id": row[0],
                "name": row[1], 
                "description": row[2],
                "created_at": row[3],
                "total_evaluated": row[4],
                "total_agents": row[5],
                "avg_overall_score": row[6],
                "data": evaluation_data,
                "metadata": metadata
            }
        
        logger.info(f"✅ Retrieved evaluation: {evaluation_id}")
        return {
            "success": True,
            "evaluation": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Get evaluation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve evaluation: {str(e)}")

@app.delete("/api/evaluations/{evaluation_id}")
async def delete_evaluation(evaluation_id: str):
    """Delete specific evaluation"""
    try:
        logger.info(f"🔄 Deleting evaluation: {evaluation_id}")
        
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Check if evaluation exists
            cursor.execute("SELECT name FROM evaluations WHERE id = ?", (evaluation_id,))
            row = cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Evaluation not found")
            
            evaluation_name = row[0]
            
            # Delete results first (foreign key constraint)
            cursor.execute("DELETE FROM evaluation_results WHERE evaluation_id = ?", (evaluation_id,))
            
            # Delete evaluation
            cursor.execute("DELETE FROM evaluations WHERE id = ?", (evaluation_id,))
            
            conn.commit()
        
        logger.info(f"✅ Deleted evaluation: {evaluation_id}")
        return {
            "success": True,
            "message": f"Evaluation '{evaluation_name}' deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Delete evaluation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete evaluation: {str(e)}")

@app.get("/api/evaluations/stats")
async def get_evaluation_stats():
    """Get statistics about stored evaluations"""
    try:
        logger.info("🔄 Fetching evaluation statistics")
        
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Basic stats
            cursor.execute("SELECT COUNT(*) FROM evaluations")
            total_evaluations = cursor.fetchone()[0]
            
            cursor.execute("SELECT SUM(total_evaluated) FROM evaluations")
            total_responses = cursor.fetchone()[0] or 0
            
            cursor.execute("SELECT AVG(avg_overall_score) FROM evaluations")
            avg_score = cursor.fetchone()[0] or 0
            
            # Recent evaluations (last 30 days)
            cursor.execute('''
                SELECT COUNT(*) FROM evaluations 
                WHERE created_at >= datetime('now', '-30 days')
            ''')
            recent_evaluations = cursor.fetchone()[0]
            
            # Top performing agents
            cursor.execute('''
                SELECT agent_id, AVG(overall_score) as avg_score, COUNT(*) as count
                FROM evaluation_results 
                GROUP BY agent_id 
                ORDER BY avg_score DESC 
                LIMIT 5
            ''')
            top_agents = [{"agent_id": row[0], "avg_score": row[1], "count": row[2]} 
                         for row in cursor.fetchall()]
        
        logger.info("✅ Retrieved evaluation statistics")
        return {
            "success": True,
            "stats": {
                "total_evaluations": total_evaluations,
                "total_responses": total_responses,
                "avg_overall_score": round(avg_score, 3) if avg_score else 0,
                "recent_evaluations": recent_evaluations,
                "top_agents": top_agents
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Get stats error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve statistics: {str(e)}")

# Keep all existing export and explanation endpoints
@app.post("/api/export/csv")
async def export_csv(request: ExportRequest):
    """Export data as CSV file"""
    try:
        logger.info("🔄 Generating CSV export...")
        data = request.data
        results = data['results']
        
        # Create main results DataFrame
        rows = []
        for result in results:
            row = {
                'prompt_id': result['prompt_id'],
                'agent_id': result['agent_id'],
                'prompt': result['prompt'],
                'response': result['response'],
                'reference': result['reference'],
                'generated_at': result['generated_at']
            }
            # Add all metrics
            for metric, score in result['metrics'].items():
                row[metric] = score
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        df.to_csv(temp_file.name, index=False)
        
        logger.info("✅ CSV export generated successfully")
        return FileResponse(
            temp_file.name,
            media_type='text/csv',
            filename=f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        
    except Exception as e:
        logger.error(f"❌ CSV export error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"CSV export failed: {str(e)}")

@app.post("/api/export/excel")
async def export_excel(request: ExportRequest):
    """Export data as Excel with multiple sheets"""
    try:
        logger.info("🔄 Generating Excel export...")
        data = request.data
        results = data['results']
        agent_summaries = data['agent_summaries']
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
        
        with pd.ExcelWriter(temp_file.name, engine='openpyxl') as writer:
            # Results sheet
            results_rows = []
            for result in results:
                row = {
                    'prompt_id': result['prompt_id'],
                    'agent_id': result['agent_id'],
                    'prompt': result['prompt'],
                    'response': result['response'],
                    'reference': result['reference'],
                    'generated_at': result['generated_at']
                }
                for metric, score in result['metrics'].items():
                    row[metric] = score
                results_rows.append(row)
            
            results_df = pd.DataFrame(results_rows)
            results_df.to_excel(writer, sheet_name='Results', index=False)
            
            # Agent summaries sheet
            summary_rows = []
            for agent_id, metrics in agent_summaries.items():
                row = {'agent_id': agent_id}
                row.update(metrics)
                summary_rows.append(row)
            
            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_excel(writer, sheet_name='Agent_Summaries', index=False)
            
            # Statistics sheet
            stats_data = {
                'Metric': ['Total Evaluated', 'Total Agents', 'Average Overall Score'],
                'Value': [
                    data['total_evaluated'],
                    data['overall_summary']['total_agents'],
                    f"{data['overall_summary']['avg_overall_score']:.3f}"
                ]
            }
            stats_df = pd.DataFrame(stats_data)
            stats_df.to_excel(writer, sheet_name='Statistics', index=False)
        
        logger.info("✅ Excel export generated successfully")
        return FileResponse(
            temp_file.name,
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            filename=f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        )
        
    except Exception as e:
        logger.error(f"❌ Excel export error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Excel export failed: {str(e)}")

@app.post("/api/export/pdf")
async def export_pdf(request: ExportRequest):
    """Export comprehensive PDF report"""
    try:
        logger.info("🔄 Generating PDF export...")
        data = request.data
        include_charts = request.include_charts
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        
        # Create PDF document
        doc = SimpleDocTemplate(temp_file.name, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        story.append(Paragraph("AI Agent Evaluation Report", title_style))
        story.append(Spacer(1, 20))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", styles['Heading2']))
        
        summary_text = f"""
        This report presents a comprehensive analysis of AI agent performance across {data['total_evaluated']} 
        evaluation instances covering {data['overall_summary']['total_agents']} different agents.
        <br/><br/>
        <b>Key Findings:</b><br/>
        • Average Overall Score: {data['overall_summary']['avg_overall_score']:.1%}<br/>
        • Total Evaluations: {data['total_evaluated']}<br/>
        • Agents Analyzed: {data['overall_summary']['total_agents']}<br/>
        • Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}<br/>
        """
        
        if request.metadata:
            summary_text += f"• Platform Version: {request.metadata.get('version', 'N/A')}<br/>"
        
        story.append(Paragraph(summary_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Agent Performance Summary Table
        story.append(Paragraph("Agent Performance Summary", styles['Heading2']))
        
        # Prepare table data
        table_data = [['Agent', 'Overall', 'Instruction', 'Accuracy', 'Coherence', 'Completeness']]
        
        for agent, metrics in data['agent_summaries'].items():
            row = [
                agent,
                f"{metrics['overall_score']:.1%}",
                f"{metrics['instruction_score']:.1%}",
                f"{metrics['accuracy_score']:.1%}",
                f"{metrics['coherence_score']:.1%}",
                f"{metrics['completeness_score']:.1%}"
            ]
            table_data.append(row)
        
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(table)
        story.append(Spacer(1, 30))
        
        # Charts (if requested)
        if include_charts:
            story.append(Paragraph("Performance Visualizations", styles['Heading2']))
            
            charts = create_visualizations(data)
            
            for chart_name, chart_b64 in charts.items():
                try:
                    # Decode base64 and create image
                    chart_data = base64.b64decode(chart_b64)
                    
                    # Create temporary image file
                    temp_img = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                    with open(temp_img.name, 'wb') as f:
                        f.write(chart_data)
                    
                    # Add image to PDF
                    img = Image(temp_img.name, width=6*inch, height=4*inch)
                    story.append(img)
                    story.append(Spacer(1, 20))
                    
                    # Clean up temp image
                    os.unlink(temp_img.name)
                    
                except Exception as chart_error:
                    logger.warning(f"⚠️ Could not add chart {chart_name}: {str(chart_error)}")
                    continue
        
        # Footer
        story.append(Spacer(1, 30))
        footer_text = f"Generated by AI Agent Evaluator v2.1 on {datetime.now().strftime('%B %d, %Y at %H:%M')}"
        story.append(Paragraph(footer_text, styles['Normal']))
        
        # Build PDF
        doc.build(story)
        
        logger.info("✅ PDF export generated successfully")
        return FileResponse(
            temp_file.name,
            media_type='application/pdf',
            filename=f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )
        
    except Exception as e:
        logger.error(f"❌ PDF export error: {str(e)}")
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"PDF export failed: {str(e)}")

# ENHANCED TRADITIONAL ENDPOINTS (with hybrid support)

@app.post("/api/upload-and-evaluate")
async def upload_and_evaluate(file: UploadFile = File(...), mode: EvaluationMode = EvaluationMode.HYBRID):
    try:
        logger.info(f"🔄 Processing upload: {file.filename} (mode: {mode})")
        
        # Read and parse file
        content = await file.read()
        data = json.loads(content.decode('utf-8'))
        
        # Validate structure
        if not isinstance(data, list) or not data:
            raise HTTPException(status_code=400, detail="JSON must be a non-empty array")
        
        results = []
        for i, item in enumerate(data):
            logger.info(f"🔄 Processing item {i+1}/{len(data)}: {item['prompt_id']}")
            
            try:
                # Ensure required fields
                required_fields = ["prompt_id", "prompt", "agent_id", "response"]
                missing_fields = [field for field in required_fields if field not in item]
                
                if missing_fields:
                    logger.warning(f"Item {i+1} missing fields: {missing_fields}")
                    continue
                
                # Get or generate reference
                reference = item.get("reference")
                ai_generated_ref = False
                
                if not reference and mode in [EvaluationMode.AI_ASSISTED, EvaluationMode.HYBRID]:
                    logger.info(f"🔄 Generating reference for item {i+1}")
                    reference = await generate_reference_answer(item["prompt"])
                    ai_generated_ref = True
                elif not reference:
                    logger.warning(f"Item {i+1} has no reference and AI generation disabled")
                    continue
                
                # Score using selected mode
                if mode == EvaluationMode.AI_ASSISTED:
                    # Use AI scoring
                    metric_scores_dict = await generate_ai_scores(
                        item["prompt"], item["response"], reference
                    )
                    metric_scores = [
                        metric_scores_dict["instruction_score"],
                        metric_scores_dict["hallucination_score"],
                        metric_scores_dict["assumption_score"],
                        metric_scores_dict["coherence_score"],
                        metric_scores_dict["accuracy_score"],
                        metric_scores_dict["completeness_score"],
                        metric_scores_dict["overall_score"]
                    ]
                elif mode == EvaluationMode.TRADITIONAL:
                    # Use traditional model
                    metric_scores = predict_metrics(item["prompt"], item["response"], reference)
                    metric_scores_dict = {
                        "instruction_score": metric_scores[0],
                        "hallucination_score": metric_scores[1],
                        "assumption_score": metric_scores[2],
                        "coherence_score": metric_scores[3],
                        "accuracy_score": metric_scores[4],
                        "completeness_score": metric_scores[5],
                        "overall_score": metric_scores[6]
                    }
                else:  # HYBRID
                    # Use both and average (or use AI as primary)
                    if openai_client:
                        ai_scores = await generate_ai_scores(item["prompt"], item["response"], reference)
                        metric_scores_dict = ai_scores
                    else:
                        traditional_scores = predict_metrics(item["prompt"], item["response"], reference)
                        metric_scores_dict = {
                            "instruction_score": traditional_scores[0],
                            "hallucination_score": traditional_scores[1],
                            "assumption_score": traditional_scores[2],
                            "coherence_score": traditional_scores[3],
                            "accuracy_score": traditional_scores[4],
                            "completeness_score": traditional_scores[5],
                            "overall_score": traditional_scores[6]
                        }
                
                result = {
                    "prompt_id": item["prompt_id"],
                    "agent_id": item["agent_id"],
                    "prompt": item["prompt"],
                    "response": item["response"],
                    "reference": reference,
                    "ai_generated_reference": ai_generated_ref,
                    "metrics": {
                        "instruction_score": round(metric_scores_dict["instruction_score"], 3),
                        "hallucination_score": round(metric_scores_dict["hallucination_score"], 3),
                        "assumption_score": round(metric_scores_dict["assumption_score"], 3),
                        "coherence_score": round(metric_scores_dict["coherence_score"], 3),
                        "accuracy_score": round(metric_scores_dict["accuracy_score"], 3),
                        "completeness_score": round(metric_scores_dict["completeness_score"], 3),
                        "overall_score": round(metric_scores_dict["overall_score"], 3),
                    },
                    "generated_at": datetime.now().isoformat(),
                    "evaluation_mode": mode
                }
                
                results.append(result)
                
            except Exception as item_error:
                logger.error(f"❌ Error processing item {i+1}: {str(item_error)}")
                continue
        
        if not results:
            raise HTTPException(status_code=500, detail="No items could be processed successfully")
        
        # Calculate summaries
        agents = {}
        for result in results:
            agent_id = result["agent_id"]
            if agent_id not in agents:
                agents[agent_id] = []
            agents[agent_id].append(result["metrics"])
        
        agent_summaries = {}
        for agent_id, metrics_list in agents.items():
            agent_summaries[agent_id] = {
                metric: round(sum(m[metric] for m in metrics_list) / len(metrics_list), 3)
                for metric in ["instruction_score", "hallucination_score", "assumption_score", 
                             "coherence_score", "accuracy_score", "completeness_score", "overall_score"]
            }
        
        response_data = {
            "success": True,
            "total_evaluated": len(results),
            "results": results,
            "agent_summaries": agent_summaries,
            "overall_summary": {
                "avg_overall_score": round(sum(r["metrics"]["overall_score"] for r in results) / len(results), 3),
                "total_agents": len(agents),
                "total_prompts": len(results)
            },
            "evaluation_mode": mode
        }
        
        logger.info(f"✅ Successfully processed {len(results)} items using {mode} mode")
        return response_data
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON decode error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")
    except Exception as e:
        logger.error(f"❌ Upload and evaluate error: {str(e)}")
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# ENHANCED EXPLANATION ENDPOINT
@app.post("/api/explain")
async def get_explanation(request: ExplanationRequest):
    """Generate AI explanation for a specific metric score (enhanced)"""
    try:
        logger.info("🔄 Generating enhanced AI explanation...")
        result = request.result
        metric = request.metric
        context = request.context or {}
        
        # Use AI-powered explanation if available
        if openai_client:
            prompt = f"""
            Provide a detailed explanation for why this specific metric received its score:

            METRIC: {metric}
            SCORE: {result['metrics'][metric]:.3f}

            PROMPT: {result['prompt']}
            AGENT RESPONSE: {result['response']}
            REFERENCE: {result['reference']}

            Context: Agent {result['agent_id']} generated this response.
            {f"Additional context: {json.dumps(context)}" if context else ""}

            Explain:
            1. What this metric measures
            2. Why this specific score was given
            3. Strengths in the response related to this metric
            4. Areas for improvement
            5. Specific examples from the response
            6. How to improve this metric in future responses

            Be specific, actionable, and educational.
            """
            
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert AI evaluator providing detailed, educational explanations about evaluation metrics and scores."
                },
                {"role": "user", "content": prompt}
            ]
            
            explanation = await call_openai_api(messages, temperature=0.5)
        else:
            # Fallback to basic explanation
            metric_descriptions = {
                'instruction_score': 'how well the response follows the given instructions and requirements',
                'hallucination_score': 'the accuracy and factual correctness of the information provided',
                'assumption_score': 'how appropriately the response handles assumptions and uncertainties',
                'coherence_score': 'the logical flow, consistency, and readability of the response',
                'accuracy_score': 'the factual correctness and precision of the information',
                'completeness_score': 'how thoroughly the response addresses all aspects of the prompt',
                'overall_score': 'the overall quality combining all evaluation dimensions'
            }
            
            score = result['metrics'][metric]
            score_percentage = score * 100
            
            if score >= 0.8:
                performance_level = "excellent"
                improvement_note = "This response demonstrates strong performance in this metric."
            elif score >= 0.6:
                performance_level = "good"
                improvement_note = "This response shows solid performance with room for minor improvements."
            elif score >= 0.4:
                performance_level = "moderate"
                improvement_note = "This response shows average performance and could benefit from targeted improvements."
            else:
                performance_level = "needs improvement"
                improvement_note = "This response shows significant room for improvement in this area."
            
            explanation = f"""The {metric.replace('_', ' ')} score of {score_percentage:.1f}% reflects {performance_level} performance in {metric_descriptions.get(metric, 'this evaluation dimension')}.

This score was calculated by analyzing the relationship between the prompt, agent response, and reference answer. {improvement_note}

Agent: {result['agent_id']}
Prompt Context: {result['prompt'][:100]}{'...' if len(result['prompt']) > 100 else ''}

The evaluation considers factors such as relevance, accuracy, completeness, and adherence to instructions when determining this score."""
            
            if context.get('overallScore'):
                explanation += f"\n\nNote: This metric contributed to an overall score of {context['overallScore'] * 100:.1f}% for this response."
        
        logger.info("✅ Enhanced AI explanation generated successfully")
        return {
            "success": True,
            "explanation": explanation,
            "metric": metric,
            "score": result['metrics'][metric],
            "generated_at": datetime.now().isoformat(),
            "ai_powered": openai_client is not None
        }
        
    except Exception as e:
        logger.error(f"❌ Explanation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate explanation: {str(e)}")

# Keep all existing evaluation history endpoints (evaluations, export, etc.)
# ... [Previous evaluation history endpoints remain the same] ...

# EVALUATION HISTORY ENDPOINTS (keeping existing ones)

@app.post("/api/evaluations/save")
async def save_evaluation(request: SaveEvaluationRequest):
    """Save evaluation results to history (enhanced with AI data)"""
    try:
        logger.info(f"🔄 Saving evaluation: {request.name}")
        
        evaluation_id = str(uuid.uuid4())
        data = request.data
        
        # Determine if this is AI-generated
        ai_generated = any(result.get("ai_generated_reference", False) for result in data.get("results", []))
        
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Save main evaluation record
            cursor.execute('''
                INSERT INTO evaluations 
                (id, name, description, total_evaluated, total_agents, avg_overall_score, data_json, metadata_json, ai_generated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                evaluation_id,
                request.name,
                request.description,
                data['total_evaluated'],
                data['overall_summary']['total_agents'],
                data['overall_summary']['avg_overall_score'],
                json.dumps(data),
                json.dumps(request.metadata) if request.metadata else None,
                ai_generated
            ))
            
            # Save individual results with AI data
            for result in data['results']:
                result_id = str(uuid.uuid4())
                cursor.execute('''
                    INSERT INTO evaluation_results 
                    (id, evaluation_id, prompt_id, agent_id, prompt, response, reference,
                     instruction_score, hallucination_score, assumption_score, coherence_score,
                     accuracy_score, completeness_score, overall_score, generated_at,
                     ai_generated_reference, ai_explanation, suggested_prompt)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    result_id, evaluation_id, result['prompt_id'], result['agent_id'],
                    result['prompt'], result['response'], result['reference'],
                    result['metrics']['instruction_score'],
                    result['metrics']['hallucination_score'],
                    result['metrics']['assumption_score'],
                    result['metrics']['coherence_score'],
                    result['metrics']['accuracy_score'],
                    result['metrics']['completeness_score'],
                    result['metrics']['overall_score'],
                    result['generated_at'],
                    result.get('ai_generated_reference', False),
                    result.get('explanation'),
                    json.dumps(result.get('prompt_suggestions')) if result.get('prompt_suggestions') else None
                ))
            
            conn.commit()
        
        logger.info(f"✅ Successfully saved evaluation: {evaluation_id}")
        return {
            "success": True,
            "evaluation_id": evaluation_id,
            "message": f"Evaluation '{request.name}' saved successfully"
        }
        
    except Exception as e:
        logger.error(f"❌ Save evaluation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save evaluation: {str(e)}")

@app.get("/api/evaluations")
async def get_evaluations(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """Get list of saved evaluations (enhanced with AI indicators)"""
    try:
        logger.info(f"🔄 Fetching evaluations (limit: {limit}, offset: {offset})")
        
        with get_db() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM evaluations")
            total_count = cursor.fetchone()[0]
            
            cursor.execute('''
                SELECT id, name, description, created_at, total_evaluated, 
                       total_agents, avg_overall_score, ai_generated
                FROM evaluations 
                ORDER BY created_at DESC 
                LIMIT ? OFFSET ?
            ''', (limit, offset))
            
            evaluations = []
            for row in cursor.fetchall():
                evaluations.append({
                    "id": row[0],
                    "name": row[1],
                    "description": row[2],
                    "created_at": row[3],
                    "total_evaluated": row[4],
                    "total_agents": row[5],
                    "avg_overall_score": row[6],
                    "ai_generated": bool(row[7])
                })
        
        logger.info(f"✅ Retrieved {len(evaluations)} evaluations")
        return {
            "success": True,
            "evaluations": evaluations,
            "pagination": {
                "total": total_count,
                "limit": limit,
                "offset": offset,
                "has_next": offset + limit < total_count
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Get evaluations error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve evaluations: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
 