import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Bot, Wand2, ArrowLeft, Plus, Trash2, Sparkles,
  Settings, ToggleLeft, ToggleRight, Lightbulb,
  FileText, MessageCircle, Target, AlertCircle,
  CheckCircle, RefreshCw, Copy, Save, 
  BarChart3, Info, Star
} from 'lucide-react';

interface AIFeatures {
  autoReference: boolean;
  detailedExplanations: boolean;
  promptSuggestions: boolean;
  assistantChat: boolean;
}

interface AIEvaluationPanelProps {
  onEvaluationComplete: (data: any) => Promise<any>;
  onBackToLanding: () => void;
  aiFeatures: AIFeatures;
  onAIFeaturesChange: (features: AIFeatures) => void;
  evaluationMode: 'traditional' | 'ai_assisted' | 'hybrid';
  onEvaluationModeChange: (mode: 'traditional' | 'ai_assisted' | 'hybrid') => void;
}

interface EvaluationItem {
  id: string;
  prompt: string;
  response: string;
  agent_id: string;
  reference?: string;
}

interface EvaluationResult {
  success: boolean;
  result?: any;
  results?: any[];
  total_evaluated?: number;
  agent_summaries?: Record<string, any>;
  overall_summary?: any;
  evaluation_id?: string;
  message?: string;
}

const AIEvaluationPanel: React.FC<AIEvaluationPanelProps> = ({
  onEvaluationComplete,
  onBackToLanding,
  aiFeatures,
  onAIFeaturesChange,
  evaluationMode,
  onEvaluationModeChange
}) => {
  const [evaluationItems, setEvaluationItems] = useState<EvaluationItem[]>([
    {
      id: '1',
      prompt: '',
      response: '',
      agent_id: '',
      reference: ''
    }
  ]);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [evaluationName, setEvaluationName] = useState('');
  const [evaluationDescription, setEvaluationDescription] = useState('');
  const [currentTab, setCurrentTab] = useState<'single' | 'bulk'>('single');
  
  // State for handling results
  const [evaluationResults, setEvaluationResults] = useState<EvaluationResult | null>(null);
  const [showResults, setShowResults] = useState(false);
  const [evaluationError, setEvaluationError] = useState<string | null>(null);

  const addEvaluationItem = () => {
    const newItem: EvaluationItem = {
      id: Date.now().toString(),
      prompt: '',
      response: '',
      agent_id: '',
      reference: ''
    };
    setEvaluationItems([...evaluationItems, newItem]);
  };

  const removeEvaluationItem = (id: string) => {
    setEvaluationItems(evaluationItems.filter(item => item.id !== id));
  };

  const updateEvaluationItem = (id: string, field: keyof EvaluationItem, value: string) => {
    setEvaluationItems(evaluationItems.map(item =>
      item.id === id ? { ...item, [field]: value } : item
    ));
  };

  const handleEvaluate = async () => {
    const validItems = evaluationItems.filter(item =>
      item.prompt.trim() && item.response.trim() && item.agent_id.trim()
    );

    if (validItems.length === 0) {
      alert('Please fill in at least one complete evaluation (prompt, response, and agent ID).');
      return;
    }

    setIsEvaluating(true);
    setEvaluationError(null);
    setEvaluationResults(null);
    setShowResults(false);

    try {
      let result: EvaluationResult;

      if (currentTab === 'single' && validItems.length === 1) {
        // Single evaluation
        console.log('🔄 Starting single evaluation...');
        result = await onEvaluationComplete({
          prompt: validItems[0].prompt,
          response: validItems[0].response,
          agent_id: validItems[0].agent_id,
          reference: validItems[0].reference || undefined,
          generate_reference: aiFeatures.autoReference,
          include_explanations: aiFeatures.detailedExplanations,
          include_suggestions: aiFeatures.promptSuggestions
        });

        console.log('✅ Single evaluation completed:', result);
      } else {
        // Bulk evaluation
        console.log('🔄 Starting bulk evaluation...');
        result = await onEvaluationComplete({
          items: validItems.map(item => ({
            prompt: item.prompt,
            response: item.response,
            agent_id: item.agent_id,
            reference: item.reference || undefined,
            generate_reference: aiFeatures.autoReference,
            include_explanations: aiFeatures.detailedExplanations,
            include_suggestions: aiFeatures.promptSuggestions
          })),
          evaluation_name: evaluationName || undefined,
          evaluation_description: evaluationDescription || undefined
        });

        console.log('✅ Bulk evaluation completed:', result);
      }

      // Store results and show them
      setEvaluationResults(result);
      setShowResults(true);

      // Success message
      const evaluationType = currentTab === 'single' ? 'Single' : 'Bulk';
      const itemCount = result.results ? result.results.length : (result.result ? 1 : 0);
      
      console.log('📊 Setting results to display:', {
        type: evaluationType,
        itemCount,
        hasResults: !!result,
        resultData: result
      });

    } catch (error) {
      console.error('❌ Evaluation failed:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      setEvaluationError(`Evaluation failed: ${errorMessage}`);
      alert(`Evaluation failed: ${errorMessage}`);
    } finally {
      setIsEvaluating(false);
    }
  };

  const toggleAIFeature = (feature: keyof AIFeatures) => {
    onAIFeaturesChange({
      ...aiFeatures,
      [feature]: !aiFeatures[feature]
    });
  };

  const resetForm = () => {
    setEvaluationItems([{
      id: '1',
      prompt: '',
      response: '',
      agent_id: '',
      reference: ''
    }]);
    setEvaluationName('');
    setEvaluationDescription('');
    setEvaluationResults(null);
    setShowResults(false);
    setEvaluationError(null);
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    alert('Copied to clipboard!');
  };

  const modeDescriptions = {
    traditional: 'Uses your trained BERT model for evaluation (faster)',
    ai_assisted: 'Powered entirely by AI for comprehensive analysis (slower but more detailed)',
    hybrid: 'Combines traditional model with AI insights (balanced approach)'
  };

  // Inline Results Display Component
  const InlineResultsDisplay = ({ results }: { results: EvaluationResult }) => {
    if (!results || !results.success) return null;

    // Handle both single and bulk results
    const resultsToShow = results.result ? [results.result] : (results.results || []);
    const isSingle = !!results.result;
    const totalEvaluated = results.total_evaluated || resultsToShow.length;

    console.log('🎯 Rendering results:', {
      isSingle,
      resultsCount: resultsToShow.length,
      totalEvaluated,
      results: resultsToShow
    });

    if (resultsToShow.length === 0) {
      return (
        <div className="mt-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
          <p className="text-yellow-800">No results to display. Please check your evaluation data.</p>
        </div>
      );
    }

    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mt-8 space-y-6"
      >
        {/* Success Header */}
        <div className="bg-gradient-to-r from-green-500 to-green-600 text-white rounded-lg p-6">
          <div className="flex items-center space-x-3">
            <CheckCircle className="w-8 h-8" />
            <div>
              <h3 className="text-xl font-bold">Evaluation Complete!</h3>
              <p className="text-green-100">
                {totalEvaluated} evaluation{totalEvaluated > 1 ? 's' : ''} processed successfully
              </p>
            </div>
          </div>
        </div>

        {/* Overall Summary (for bulk only) */}
        {!isSingle && results.overall_summary && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-blue-50 rounded-lg p-4 text-center border border-blue-200">
              <div className="text-2xl font-bold text-blue-600">{totalEvaluated}</div>
              <div className="text-sm text-blue-700">Total Evaluated</div>
            </div>
            <div className="bg-purple-50 rounded-lg p-4 text-center border border-purple-200">
              <div className="text-2xl font-bold text-purple-600">
                {results.overall_summary.total_agents || 0}
              </div>
              <div className="text-sm text-purple-700">Agents</div>
            </div>
            <div className="bg-green-50 rounded-lg p-4 text-center border border-green-200">
              <div className="text-2xl font-bold text-green-600">
                {((results.overall_summary.avg_overall_score || 0) * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-green-700">Avg Score</div>
            </div>
            <div className="bg-indigo-50 rounded-lg p-4 text-center border border-indigo-200">
              <div className="text-2xl font-bold text-indigo-600">
                {results.evaluation_id ? 'Saved' : 'Complete'}
              </div>
              <div className="text-sm text-indigo-700">Status</div>
            </div>
          </div>
        )}

        {/* Individual Results */}
        <div className="space-y-6">
          {resultsToShow.map((result, index) => (
            <div key={result.prompt_id || index} className="bg-white rounded-lg shadow-lg border border-gray-200 overflow-hidden">
              {/* Result Header */}
              <div className="bg-gray-50 px-6 py-4 border-b border-gray-200">
                <div className="flex items-center justify-between">
                  <h4 className="text-lg font-semibold text-gray-900">
                    Agent: {result.agent_id}
                  </h4>
                  <div className="flex items-center space-x-3">
                    <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                      (result.metrics?.overall_score || 0) >= 0.8 
                        ? 'bg-green-100 text-green-800'
                        : (result.metrics?.overall_score || 0) >= 0.6
                        ? 'bg-yellow-100 text-yellow-800'
                        : 'bg-red-100 text-red-800'
                    }`}>
                      {((result.metrics?.overall_score || 0) * 100).toFixed(1)}% Overall Score
                    </div>
                    {result.ai_generated_reference && (
                      <div className="flex items-center text-xs text-blue-600 bg-blue-50 px-2 py-1 rounded">
                        <Sparkles className="w-3 h-3 mr-1" />
                        AI Reference
                      </div>
                    )}
                  </div>
                </div>
              </div>

              <div className="p-6 space-y-6">
                {/* Metrics Grid */}
                {result.metrics && (
                  <div>
                    <h5 className="font-medium text-gray-900 mb-4 flex items-center">
                      <Target className="w-5 h-5 mr-2" />
                      Evaluation Metrics
                    </h5>
                    <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3">
                      {[
                        { key: 'instruction_score', label: 'Instruction Following' },
                        { key: 'hallucination_score', label: 'Accuracy' },
                        { key: 'assumption_score', label: 'Assumptions' },
                        { key: 'coherence_score', label: 'Coherence' },
                        { key: 'accuracy_score', label: 'Precision' },
                        { key: 'completeness_score', label: 'Completeness' },
                        { key: 'overall_score', label: 'Overall' }
                      ].map(({ key, label }) => {
                        const score = result.metrics[key] || 0;
                        const percentage = (score * 100).toFixed(1);
                        return (
                          <div key={key} className={`p-3 rounded-lg text-center border ${
                            score >= 0.8 ? 'bg-green-50 border-green-200' :
                            score >= 0.6 ? 'bg-yellow-50 border-yellow-200' : 
                            'bg-red-50 border-red-200'
                          }`}>
                            <div className={`text-lg font-bold ${
                              score >= 0.8 ? 'text-green-600' :
                              score >= 0.6 ? 'text-yellow-600' : 'text-red-600'
                            }`}>
                              {percentage}%
                            </div>
                            <div className="text-xs text-gray-600 mt-1">{label}</div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* Content Sections */}
                <div className="space-y-4">
                  {/* Prompt */}
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <h6 className="font-medium text-gray-900 flex items-center">
                        <MessageCircle className="w-4 h-4 mr-2" />
                        Prompt
                      </h6>
                      <button
                        onClick={() => copyToClipboard(result.prompt)}
                        className="text-gray-500 hover:text-gray-700 p-1 rounded"
                        title="Copy prompt"
                      >
                        <Copy className="w-4 h-4" />
                      </button>
                    </div>
                    <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                      <p className="text-sm text-gray-800 whitespace-pre-wrap">{result.prompt}</p>
                    </div>
                  </div>

                  {/* Response */}
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <h6 className="font-medium text-gray-900 flex items-center">
                        <Bot className="w-4 h-4 mr-2" />
                        Agent Response
                      </h6>
                      <button
                        onClick={() => copyToClipboard(result.response)}
                        className="text-gray-500 hover:text-gray-700 p-1 rounded"
                        title="Copy response"
                      >
                        <Copy className="w-4 h-4" />
                      </button>
                    </div>
                    <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
                      <p className="text-sm text-gray-800 whitespace-pre-wrap">{result.response}</p>
                    </div>
                  </div>

                  {/* Reference */}
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <h6 className="font-medium text-gray-900 flex items-center">
                        <Star className="w-4 h-4 mr-2" />
                        Reference Answer
                        {result.ai_generated_reference && (
                          <span className="ml-2 text-xs text-blue-600 bg-blue-100 px-2 py-1 rounded">
                            AI Generated
                          </span>
                        )}
                      </h6>
                      <button
                        onClick={() => copyToClipboard(result.reference)}
                        className="text-gray-500 hover:text-gray-700 p-1 rounded"
                        title="Copy reference"
                      >
                        <Copy className="w-4 h-4" />
                      </button>
                    </div>
                    <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                      <p className="text-sm text-gray-800 whitespace-pre-wrap">{result.reference}</p>
                    </div>
                  </div>
                </div>

                {/* AI Explanation */}
                {result.explanation && aiFeatures.detailedExplanations && (
                  <div>
                    <h6 className="font-medium text-gray-900 mb-3 flex items-center">
                      <Info className="w-4 h-4 mr-2" />
                      AI Explanation
                    </h6>
                    <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-4">
                      <p className="text-sm text-gray-800 whitespace-pre-wrap">{result.explanation}</p>
                    </div>
                  </div>
                )}

                {/* Prompt Suggestions */}
                {result.prompt_suggestions && aiFeatures.promptSuggestions && (
                  <div>
                    <h6 className="font-medium text-gray-900 mb-3 flex items-center">
                      <Lightbulb className="w-4 h-4 mr-2" />
                      Prompt Improvement Suggestions
                    </h6>
                    <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 space-y-3">
                      {typeof result.prompt_suggestions === 'object' ? (
                        <>
                          {result.prompt_suggestions.analysis && (
                            <div>
                              <h6 className="font-medium text-sm text-gray-900">Analysis:</h6>
                              <p className="text-sm text-gray-700 mt-1">{result.prompt_suggestions.analysis}</p>
                            </div>
                          )}
                          
                          {result.prompt_suggestions.suggestions && Array.isArray(result.prompt_suggestions.suggestions) && (
                            <div>
                              <h7 className="font-medium text-sm text-gray-900">Suggestions:</h7>
                              <ul className="text-sm text-gray-700 mt-1 space-y-1">
                                {result.prompt_suggestions.suggestions.map((suggestion: string, idx: number) => (
                                  <li key={idx} className="flex items-start">
                                    <span className="text-yellow-600 mr-2">•</span>
                                    {suggestion}
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}
                          
                          {result.prompt_suggestions.improved_prompt && (
                            <div>
                              <div className="flex items-center justify-between mb-2">
                                <h7 className="font-medium text-sm text-gray-900">Improved Prompt:</h7>
                                <button
                                  onClick={() => copyToClipboard(result.prompt_suggestions.improved_prompt)}
                                  className="text-gray-500 hover:text-gray-700 p-1 rounded"
                                  title="Copy improved prompt"
                                >
                                  <Copy className="w-3 h-3" />
                                </button>
                              </div>
                              <div className="bg-white border border-yellow-300 rounded p-3">
                                <p className="text-sm text-gray-800 whitespace-pre-wrap">
                                  {result.prompt_suggestions.improved_prompt}
                                </p>
                              </div>
                            </div>
                          )}
                        </>
                      ) : (
                        <p className="text-sm text-gray-700">{result.prompt_suggestions}</p>
                      )}
                    </div>
                  </div>
                )}

                {/* Metadata */}
                <div className="pt-4 border-t border-gray-200">
                  <div className="flex items-center justify-between text-xs text-gray-500">
                    <span>Evaluation ID: {result.prompt_id}</span>
                    <span>Generated: {new Date(result.generated_at).toLocaleString()}</span>
                    <span>Mode: {result.evaluation_mode || evaluationMode}</span>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Action Buttons */}
        <div className="flex flex-wrap gap-3 pt-4">
          <button
            onClick={() => onBackToLanding()}
            className="flex items-center space-x-2 px-6 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
          >
            <BarChart3 className="w-4 h-4" />
            <span>View Full Dashboard</span>
          </button>
          
          <button
            onClick={resetForm}
            className="flex items-center space-x-2 px-6 py-3 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
          >
            <Plus className="w-4 h-4" />
            <span>New Evaluation</span>
          </button>

          {results.evaluation_id && (
            <div className="flex items-center space-x-2 px-4 py-2 bg-green-100 text-green-700 rounded-lg">
              <Save className="w-4 h-4" />
              <span className="text-sm">Auto-saved</span>
            </div>
          )}
        </div>
      </motion.div>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-indigo-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <button
            onClick={onBackToLanding}
            className="flex items-center space-x-2 text-gray-600 hover:text-gray-900 transition-colors"
          >
            <ArrowLeft className="w-5 h-5" />
            <span>Back</span>
          </button>
          
          <div className="text-center">
            <h1 className="text-3xl font-bold text-gray-900 flex items-center justify-center space-x-3">
              <Bot className="w-8 h-8 text-indigo-600" />
              <span>AI-Powered Evaluation</span>
            </h1>
            <p className="text-gray-600 mt-2">Intelligent evaluation with automated reference generation and explanations</p>
          </div>
          
          <div className="w-20"></div>
        </div>

        {/* Error Display */}
        {evaluationError && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
            <div className="flex items-center space-x-2">
              <AlertCircle className="w-5 h-5 text-red-600" />
              <span className="text-red-800 font-medium">Evaluation Error</span>
            </div>
            <p className="text-red-700 mt-1">{evaluationError}</p>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Settings Panel */}
          <div className="space-y-6">
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <Settings className="w-5 h-5 mr-2" />
                AI Settings
              </h3>

              {/* Evaluation Mode */}
              <div className="mb-6">
                <h4 className="font-medium text-gray-900 mb-3">Evaluation Mode</h4>
                <div className="space-y-3">
                  {(['traditional', 'ai_assisted', 'hybrid'] as const).map(mode => (
                    <label key={mode} className="flex items-start space-x-3 p-3 rounded-lg border hover:bg-gray-50 cursor-pointer">
                      <input
                        type="radio"
                        name="evaluationMode"
                        value={mode}
                        checked={evaluationMode === mode}
                        onChange={(e) => onEvaluationModeChange(e.target.value as any)}
                        className="mt-1 text-indigo-600 border-gray-300 focus:ring-indigo-500"
                      />
                      <div>
                        <div className="font-medium text-gray-900 capitalize">
                          {mode.replace('_', ' ')}
                        </div>
                        <div className="text-sm text-gray-600">
                          {modeDescriptions[mode]}
                        </div>
                      </div>
                    </label>
                  ))}
                </div>
              </div>

              {/* AI Features */}
              <div>
                <h4 className="font-medium text-gray-900 mb-3">AI Features</h4>
                <div className="space-y-3">
                  {[
                    { key: 'autoReference', label: 'Auto Reference', icon: FileText },
                    { key: 'detailedExplanations', label: 'Explanations', icon: MessageCircle },
                    { key: 'promptSuggestions', label: 'Prompt Tips', icon: Lightbulb }
                  ].map(({ key, label, icon: Icon }) => (
                    <div key={key} className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <Icon className="w-4 h-4 text-gray-600" />
                        <span className="text-gray-900">{label}</span>
                      </div>
                      <button
                        onClick={() => toggleAIFeature(key as keyof AIFeatures)}
                        className={`p-1 rounded-full transition-colors ${
                          aiFeatures[key as keyof AIFeatures] ? 'text-green-600' : 'text-gray-400'
                        }`}
                      >
                        {aiFeatures[key as keyof AIFeatures] ? (
                          <ToggleRight className="w-6 h-6" />
                        ) : (
                          <ToggleLeft className="w-6 h-6" />
                        )}
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Info Panel */}
            <div className="bg-gradient-to-br from-indigo-50 to-purple-50 rounded-xl p-6 border border-indigo-200">
              <h3 className="font-semibold text-gray-900 mb-3 flex items-center">
                <Sparkles className="w-5 h-5 mr-2 text-indigo-600" />
                AI-Enhanced Evaluation
              </h3>
              <div className="space-y-2 text-sm text-gray-700">
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-indigo-400 rounded-full"></div>
                  <span>Automatic reference generation</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-indigo-400 rounded-full"></div>
                  <span>Detailed score explanations</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-indigo-400 rounded-full"></div>
                  <span>Prompt improvement suggestions</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-indigo-400 rounded-full"></div>
                  <span>Multi-modal evaluation support</span>
                </div>
              </div>
            </div>
          </div>

          {/* Main Content */}
          <div className="lg:col-span-3 space-y-6">
            <div className="bg-white rounded-xl shadow-lg overflow-hidden">
              {/* Tabs */}
              <div className="flex items-center space-x-1 p-6 pb-0">
                <button
                  onClick={() => setCurrentTab('single')}
                  className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                    currentTab === 'single'
                      ? 'bg-indigo-100 text-indigo-700'
                      : 'text-gray-500 hover:text-gray-700'
                  }`}
                >
                  Single Evaluation
                </button>
                <button
                  onClick={() => setCurrentTab('bulk')}
                  className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                    currentTab === 'bulk'
                      ? 'bg-indigo-100 text-indigo-700'
                      : 'text-gray-500 hover:text-gray-700'
                  }`}
                >
                  Bulk Evaluation
                </button>
              </div>

              <div className="p-6 space-y-6">
                {/* Bulk Evaluation Settings */}
                {currentTab === 'bulk' && (
                  <div className="bg-gray-50 rounded-lg p-4 space-y-4">
                    <h3 className="font-medium text-gray-900">Evaluation Details</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <input
                        type="text"
                        placeholder="Evaluation name (optional)"
                        value={evaluationName}
                        onChange={(e) => setEvaluationName(e.target.value)}
                        className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                      />
                      <textarea
                        placeholder="Description (optional)"
                        value={evaluationDescription}
                        onChange={(e) => setEvaluationDescription(e.target.value)}
                        className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                        rows={1}
                      />
                    </div>
                  </div>
                )}

                {/* Evaluation Items */}
                <div className="space-y-4">
                  {evaluationItems.map((item, index) => (
                    <motion.div
                      key={item.id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="bg-gray-50 rounded-lg p-4 space-y-4"
                    >
                      <div className="flex items-center justify-between">
                        <h3 className="font-medium text-gray-900">Evaluation {index + 1}</h3>
                        {evaluationItems.length > 1 && (
                          <button
                            onClick={() => removeEvaluationItem(item.id)}
                            className="p-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        )}
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-1">Agent ID *</label>
                          <input
                            type="text"
                            placeholder="e.g., gpt-4, claude-3, etc."
                            value={item.agent_id}
                            onChange={(e) => updateEvaluationItem(item.id, 'agent_id', e.target.value)}
                            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                          />
                        </div>

                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-1">
                            Reference Answer {aiFeatures.autoReference && '(Optional - AI will generate)'}
                          </label>
                          <input
                            type="text"
                            placeholder={aiFeatures.autoReference ? "Leave empty for AI generation" : "Expected/ideal answer"}
                            value={item.reference}
                            onChange={(e) => updateEvaluationItem(item.id, 'reference', e.target.value)}
                            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                          />
                        </div>
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Prompt *</label>
                        <textarea
                          placeholder="Enter the prompt/question given to the AI agent..."
                          value={item.prompt}
                          onChange={(e) => updateEvaluationItem(item.id, 'prompt', e.target.value)}
                          rows={3}
                          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 resize-none"
                        />
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Agent Response *</label>
                        <textarea
                          placeholder="Enter the response generated by the AI agent..."
                          value={item.response}
                          onChange={(e) => updateEvaluationItem(item.id, 'response', e.target.value)}
                          rows={4}
                          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 resize-none"
                        />
                      </div>
                    </motion.div>
                  ))}
                </div>

                {/* Add Item Button */}
                {currentTab === 'bulk' && (
                  <button
                    onClick={addEvaluationItem}
                    className="w-full flex items-center justify-center space-x-2 py-3 border-2 border-dashed border-gray-300 rounded-lg text-gray-600 hover:border-indigo-300 hover:text-indigo-600 transition-colors"
                  >
                    <Plus className="w-5 h-5" />
                    <span>Add Another Evaluation</span>
                  </button>
                )}

                {/* Evaluate Button */}
                <button
                  onClick={handleEvaluate}
                  disabled={isEvaluating}
                  className="w-full flex items-center justify-center space-x-3 py-4 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-lg hover:from-indigo-700 hover:to-purple-700 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed font-medium text-lg shadow-lg hover:shadow-xl"
                >
                  {isEvaluating ? (
                    <>
                      <RefreshCw className="w-5 h-5 animate-spin" />
                      <span>Processing with AI...</span>
                    </>
                  ) : (
                    <>
                      <Wand2 className="w-5 h-5" />
                      <span>Evaluate with AI</span>
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* INLINE RESULTS DISPLAY - Shows below the evaluation form */}
            <AnimatePresence>
              {showResults && evaluationResults && (
                <InlineResultsDisplay results={evaluationResults} />
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AIEvaluationPanel;