import React, { useState, useCallback, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Upload,FileText, AlertTriangle, CheckCircle, 
  Zap, Brain, Shield, ChevronRight, X, RefreshCw,
   MessageCircle, HelpCircle, History, Bot, Lightbulb, Wand2, Sparkles
} from 'lucide-react';
import UploadPanel from './components/UploadPanel';
import ComprehensiveDashboard from './components/ComprehensiveDashboard';
import EvaluationHistory from './components/EvaluationHistory';
import SaveEvaluationModal from './components/SaveEvaluationModal';
import AIAssistant from './components/AIAssistant';
import AIEvaluationPanel from './components/AIEvaluationPanel';
import AnimatedCounter from './components/AnimatedCounter';
 
// Enhanced Types
interface Metrics {
  instruction_score: number;
  hallucination_score: number;
  assumption_score: number;
  coherence_score: number;
  accuracy_score: number;
  completeness_score: number;
  overall_score: number;
}

interface EvaluationResult {
  prompt_id: string;
  agent_id: string;
  prompt: string;
  response: string;
  reference: string;
  ai_generated_reference?: boolean;
  metrics: Metrics;
  generated_at: string;
  explanation?: string;
  prompt_suggestions?: {
    analysis: string;
    suggestions: string[];
    improved_prompt: string;
    expected_improvements: string;
  };
  evaluation_mode?: 'traditional' | 'ai_assisted' | 'hybrid';
}

interface DashboardData {
  success: boolean;
  total_evaluated: number;
  results: EvaluationResult[];
  agent_summaries: Record<string, Metrics>;
  overall_summary: {
    avg_overall_score: number;
    total_agents: number;
    total_prompts: number;
  };
  evaluation_mode?: 'traditional' | 'ai_assisted' | 'hybrid';
}


interface NotificationState {
  message: string;
  type: 'success' | 'error' | 'info' | 'warning';
  duration?: number;
}

interface AIMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  context?: any;
}

// Main App Component
const App: React.FC = () => {
  // Core State Management
  const [currentView, setCurrentView] = useState<'landing' | 'upload' | 'dashboard' | 'history' | 'ai-eval'>('landing');
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null);
  const [currentEvaluationId, setCurrentEvaluationId] = useState<string | null>(null);
  const [isUploadOpen, setIsUploadOpen] = useState(false);
  const [isSaveModalOpen, setIsSaveModalOpen] = useState(false);
  const [notification, setNotification] = useState<NotificationState | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState('');
  const [apiHealth, setApiHealth] = useState<'checking' | 'healthy' | 'unhealthy'>('checking');
  const [evaluationStats, setEvaluationStats] = useState<any>(null);

  const API_BASE_URL="http://127.0.0.1:8000"
  // AI Features State
  const [isAIAssistantOpen, setIsAIAssistantOpen] = useState(false);
  const [aiMessages, setAIMessages] = useState<AIMessage[]>([
    {
      id: '1',
      role: 'system',
      content: 'Hello! I\'m your AI Assistant. I can help you understand evaluation metrics, navigate the dashboard, and improve your AI agents. What would you like to know?',
      timestamp: new Date().toISOString()
    }
  ]);
  const [isAITyping, setIsAITyping] = useState(false);
  const [aiSessionId] = useState(() => `session_${Date.now()}`);
  const [evaluationMode, setEvaluationMode] = useState<'traditional' | 'ai_assisted' | 'hybrid'>('hybrid');
  const [aiFeatures, setAIFeatures] = useState({
    autoReference: true,
    detailedExplanations: true,
    promptSuggestions: true,
    assistantChat: true
  });

  // Refs
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // API Health Check
  useEffect(() => {
    checkApiHealth();
    fetchEvaluationStats();
    // Auto-refresh health every 30 seconds
    const healthInterval = setInterval(checkApiHealth, 30000);
    return () => clearInterval(healthInterval);
  }, []);

  // Auto-scroll chat messages
  useEffect(() => {
    scrollToBottom();
  }, [aiMessages, isAITyping]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const checkApiHealth = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/health', {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (response.ok) {
        const healthData = await response.json();
        setApiHealth(
          (healthData.traditional_model_loaded || healthData.ai_assistant_available) ? 'healthy' : 'unhealthy'
        );
      } else {
        setApiHealth('unhealthy');
      }
    } catch (error) {
      console.error('Health check failed:', error);
      setApiHealth('unhealthy');
    }
  };

  const fetchEvaluationStats = useCallback(async () => {
    try {
      setIsLoading(true);
      const response = await fetch(`${API_BASE_URL}/api/evaluations/stats`);
      const data = await response.json();
      
      if (data.success) {
        setEvaluationStats(data.stats);
      }
    } catch (error) {
      console.error('Failed to fetch stats:', error);
    } finally {
      setIsLoading(false);
    }
  }, [API_BASE_URL]);

  // Notification Management
  const showNotification = useCallback((notification: NotificationState) => {
    setNotification(notification);
    const duration = notification.duration || (notification.type === 'error' ? 8000 : 5000);
    setTimeout(() => setNotification(null), duration);
  }, []);

  // Upload Handlers
  const handleUploadComplete = useCallback((result: DashboardData) => {
    console.log('Upload completed:', result);
    setDashboardData(result);
    setCurrentEvaluationId(null);
    setCurrentView('dashboard');
    setIsUploadOpen(false);
    
    const modeText = result.evaluation_mode === 'ai_assisted' ? '(AI-Assisted)' : 
                    result.evaluation_mode === 'hybrid' ? '(Hybrid Mode)' : 
                    '(Traditional)';
    
    showNotification({
      message: `🎉 Successfully evaluated ${result.total_evaluated} responses from ${result.overall_summary.total_agents} agents! Average score: ${(result.overall_summary.avg_overall_score * 100).toFixed(1)}% ${modeText}`,
      type: 'success'
    });
  }, [showNotification]);

  const handleUploadError = useCallback((error: string) => {
    console.error('Upload error:', error);
    showNotification({
      message: `❌ Upload failed: ${error}`,
      type: 'error'
    });
  }, [showNotification]);

  // AI Assistant Functions
  const sendAIMessage = useCallback(async (message: string) => {
    const userMessage: AIMessage = {
      id: `user_${Date.now()}`,
      role: 'user',
      content: message,
      timestamp: new Date().toISOString()
    };
    
    setAIMessages(prev => [...prev, userMessage]);
    setIsAITyping(true);
    
    try {
      const response = await fetch('http://127.0.0.1:8000/api/ai-assistant/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message,
          session_id: aiSessionId,
          context: {
            current_view: currentView,
            has_data: !!dashboardData,
            evaluation_mode: evaluationMode,
            total_evaluations: dashboardData?.total_evaluated || 0,
            ai_features_enabled: aiFeatures
          },
          evaluation_id: currentEvaluationId
        })
      });
      
      if (!response.ok) {
        throw new Error(`AI Assistant error: ${response.statusText}`);
      }
      
      const result = await response.json();
      
      const aiMessage: AIMessage = {
        id: `ai_${Date.now()}`,
        role: 'assistant',
        content: result.response,
        timestamp: new Date().toISOString(),
        context: result.context
      };
      
      setAIMessages(prev => [...prev, aiMessage]);
      
    } catch (error) {
      console.error('AI Assistant error:', error);
      const errorMessage: AIMessage = {
        id: `error_${Date.now()}`,
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please make sure the backend is running with OpenAI API configured, or try again later.',
        timestamp: new Date().toISOString()
      };
      setAIMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsAITyping(false);
    }
  }, [aiSessionId, currentView, dashboardData, evaluationMode, aiFeatures, currentEvaluationId]);

  // AI Evaluation Functions - UPDATED to handle both single and bulk results properly
  const handleAIEvaluation = useCallback(async (evaluationData: any) => {
    setIsLoading(true);
    setLoadingMessage('Processing with AI...');
    
    try {
      const isBulk = Array.isArray(evaluationData.items);
      const endpoint = isBulk ? '/api/ai-evaluation/bulk' : '/api/ai-evaluation/single';
      
      const payload = isBulk ? evaluationData : {
        ...evaluationData,
        generate_reference: aiFeatures.autoReference,
        include_explanations: aiFeatures.detailedExplanations,
        include_suggestions: aiFeatures.promptSuggestions
      };
      
      const response = await fetch(`http://127.0.0.1:8000${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(`AI Evaluation failed: ${response.statusText} - ${errorData.detail || 'Unknown error'}`);
      }
      
      const result = await response.json();
      
      if (isBulk) {
        // Handle bulk evaluation result - ensure proper DashboardData format
        const dashboardData: DashboardData = {
          success: result.success,
          total_evaluated: result.total_evaluated || result.results?.length || 0,
          results: result.results || [],
          agent_summaries: result.agent_summaries || {},
          overall_summary: result.overall_summary || {
            avg_overall_score: result.results?.reduce((sum: number, r: EvaluationResult) => sum + (r.metrics?.overall_score || 0), 0) / (result.results?.length || 1) || 0,
            total_agents: new Set(result.results?.map((r: EvaluationResult) => r.agent_id)).size || 0,
            total_prompts: result.results?.length || 0
          },
          evaluation_mode: evaluationMode
        };
        
        setDashboardData(dashboardData);
        setCurrentView('dashboard');
        showNotification({
          message: `🤖 AI evaluated ${dashboardData.total_evaluated} responses successfully!`,
          type: 'success'
        });
        return dashboardData;
      } else {
        // Handle single evaluation result - convert to DashboardData format
        const singleResult = result.result || result;
        const dashboardData: DashboardData = {
          success: true,
          total_evaluated: 1,
          results: [singleResult],
          agent_summaries: {
            [singleResult.agent_id]: singleResult.metrics
          },
          overall_summary: {
            avg_overall_score: singleResult.metrics.overall_score,
            total_agents: 1,
            total_prompts: 1
          },
          evaluation_mode: evaluationMode
        };
        
        setDashboardData(dashboardData);
        setCurrentView('dashboard');
        showNotification({
          message: `✨ AI evaluation completed! Overall score: ${(singleResult.metrics.overall_score * 100).toFixed(1)}%`,
          type: 'success'
        });
        return dashboardData;
      }
      
    } catch (error) {
      console.error('AI Evaluation error:', error);
      showNotification({
        message: `❌ AI Evaluation failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
        type: 'error'
      });
      throw error; // Re-throw to let the AIEvaluationPanel handle it
    } finally {
      setIsLoading(false);
      setLoadingMessage('');
    }
  }, [aiFeatures, evaluationMode, showNotification]);

  // Save Evaluation
  const handleSaveEvaluation = useCallback(async (name: string, description?: string) => {
    if (!dashboardData) return;
    
    setIsLoading(true);
    setLoadingMessage('Saving evaluation...');
    
    try {
      const response = await fetch('http://127.0.0.1:8000/api/evaluations/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data: dashboardData,
          name,
          description,
          metadata: {
            version: '3.0.0',
            ai_features: aiFeatures,
            evaluation_mode: dashboardData.evaluation_mode,
            saved_at: new Date().toISOString()
          }
        })
      });
      
      if (!response.ok) {
        throw new Error(`Failed to save evaluation: ${response.statusText}`);
      }
      
      const result = await response.json();
      setCurrentEvaluationId(result.evaluation_id);
      setIsSaveModalOpen(false);
      
      showNotification({
        message: `✅ Evaluation "${name}" saved successfully!`,
        type: 'success'
      });
      
      fetchEvaluationStats();
      
    } catch (error) {
      console.error('Save failed:', error);
      showNotification({
        message: `❌ Failed to save evaluation: ${error instanceof Error ? error.message : 'Unknown error'}`,
        type: 'error'
      });
    } finally {
      setIsLoading(false);
      setLoadingMessage('');
    }
  }, [dashboardData, aiFeatures, showNotification, fetchEvaluationStats]);

  // Load Evaluation from History
  const handleLoadEvaluation = useCallback(async (evaluationId: string) => {
    setIsLoading(true);
    setLoadingMessage('Loading evaluation...');
    
    try {
      const response = await fetch(`http://127.0.0.1:8000/api/evaluations/${evaluationId}`);
      
      if (!response.ok) {
        throw new Error(`Failed to load evaluation: ${response.statusText}`);
      }
      
      const result = await response.json();
      setDashboardData(result.evaluation.data);
      setCurrentEvaluationId(evaluationId);
      setCurrentView('dashboard');
      
      showNotification({
        message: `📊 Loaded evaluation "${result.evaluation.name}"`,
        type: 'success'
      });
      
    } catch (error) {
      console.error('Load failed:', error);
      showNotification({
        message: `❌ Failed to load evaluation: ${error instanceof Error ? error.message : 'Unknown error'}`,
        type: 'error'
      });
    } finally {
      setIsLoading(false);
      setLoadingMessage('');
    }
  }, [showNotification]);

  // Export Functionality
  const handleExport = useCallback(async (type: 'pdf' | 'excel' | 'csv') => {
    if (!dashboardData) {
      showNotification({
        message: 'No data available for export',
        type: 'warning'
      });
      return;
    }
    
    setIsLoading(true);
    setLoadingMessage(`Generating ${type.toUpperCase()} report...`);
    
    try {
      const exportData = {
        data: dashboardData,
        export_type: type,
        include_charts: true,
        timestamp: new Date().toISOString(),
        metadata: {
          totalEvaluations: dashboardData.total_evaluated,
          totalAgents: dashboardData.overall_summary.total_agents,
          averageScore: dashboardData.overall_summary.avg_overall_score,
          exportedBy: 'AI Agent Evaluator Dashboard',
          version: '3.0.0',
          evaluationId: currentEvaluationId,
          evaluationMode: dashboardData.evaluation_mode,
          aiFeatures: aiFeatures
        }
      };

      const response = await fetch(`http://127.0.0.1:8000/api/export/${type}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(exportData)
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(`Export failed: ${response.statusText} - ${errorData.detail || 'Unknown error'}`);
      }
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `ai-evaluation-report-${new Date().toISOString().split('T')[0]}.${type}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
      
      showNotification({
        message: `✅ Successfully exported ${type.toUpperCase()} report! Check your downloads folder.`,
        type: 'success'
      });
    } catch (error) {
      console.error('Export failed:', error);
      showNotification({
        message: `❌ Export failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
        type: 'error'
      });
    } finally {
      setIsLoading(false);
      setLoadingMessage('');
    }
  }, [dashboardData, currentEvaluationId, aiFeatures, showNotification]);

  // AI Explanation Functionality
  const handleGetExplanation = useCallback(async (result: EvaluationResult, metric: string): Promise<string> => {
    try {
      const response = await fetch('http://127.0.0.1:8000/api/explain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          result, 
          metric,
          context: {
            agentId: result.agent_id,
            promptId: result.prompt_id,
            overallScore: result.metrics.overall_score,
            timestamp: result.generated_at,
            metricScore: result.metrics[metric as keyof Metrics],
            evaluationId: currentEvaluationId,
            evaluationMode: result.evaluation_mode,
            aiGenerated: result.ai_generated_reference
          }
        })
      });
      
      if (!response.ok) {
        throw new Error(`Failed to get explanation: ${response.statusText}`);
      }
      
      const data = await response.json();
      return data.explanation || 'Explanation not available at this time.';
    } catch (error) {
      console.error('Explanation failed:', error);
      return `Failed to generate AI explanation: ${error instanceof Error ? error.message : 'Unknown error'}. Please try again later.`;
    }
  }, [currentEvaluationId]);

  // Navigation Functions
  const handleBackToLanding = useCallback(() => {
    setCurrentView('landing');
    setDashboardData(null);
    setCurrentEvaluationId(null);
  }, []);

  const handleNewEvaluation = useCallback(() => {
    setIsUploadOpen(true);
  }, []);

  const handleViewHistory = useCallback(() => {
    setCurrentView('history');
  }, []);

  const handleAIEvaluationMode = useCallback(() => {
    setCurrentView('ai-eval');
  }, []);

  // Enhanced statistics with AI features
  const stats = [
    {
      icon: FileText,
      label: 'AI-Powered Metrics',
      value: '7',
      suffix: ' Metrics',
      description: 'Comprehensive evaluation including instruction following, accuracy, coherence, and hallucination detection',
      color: 'from-blue-500 to-blue-600',
      delay: 0
    },
    {
      icon: Brain,
      label: 'AI Assistant Features',
      value: '5',
      suffix: '',
      description: 'Reference generation, detailed explanations, prompt improvements, and interactive chat support',
      color: 'from-purple-500 to-purple-600',
      delay: 0.1
    },
    {
      icon: History,
      label: 'Evaluation History',
      value: evaluationStats?.total_evaluations?.toString() || '3',
      suffix: ' Saved',
      description: 'Store and revisit previous evaluations with full AI-enhanced dashboard functionality',
      color: 'from-green-500 to-green-600',
      delay: 0.1
    },
    {
      icon: Wand2,
      label: 'Evaluation Modes',
      value: '3',
      suffix: ' Modes',
      description: 'Traditional, AI-Assisted, and Hybrid evaluation modes for maximum flexibility',
      color: 'from-yellow-500 to-yellow-600',
      delay: 0.3
    }
  ];

  // Render based on current view
  if (currentView === 'dashboard' && dashboardData) {
    return (
      <div className="relative min-h-screen">
        <ComprehensiveDashboard
          data={dashboardData}
          onExport={handleExport}
          onGetExplanation={handleGetExplanation}
          evaluationId={currentEvaluationId}
          onSave={currentEvaluationId ? undefined : (() => setIsSaveModalOpen(true))}
          aiFeatures={aiFeatures}
          evaluationMode={dashboardData.evaluation_mode}
        />
        
        {/* Navigation Buttons */}
        <div className="fixed bottom-6 left-6 flex space-x-3">
          <motion.button
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={handleBackToLanding}
            className="flex items-center space-x-2 px-4 py-3 bg-white/90 backdrop-blur-sm border border-gray-200 rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 text-gray-700 hover:text-gray-900 hover:border-indigo-300"
          >
            <ChevronRight className="w-4 h-4 rotate-180" />
            <span className="text-sm font-medium">Back</span>
          </motion.button>
          
          <motion.button
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={handleViewHistory}
            className="flex items-center space-x-2 px-4 py-3 bg-white/90 backdrop-blur-sm border border-gray-200 rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 text-gray-700 hover:text-gray-900 hover:border-purple-300"
          >
            <History className="w-4 h-4" />
            <span className="text-sm font-medium">History</span>
          </motion.button>
        </div>
        {/* AI Assistant Button */}
        <motion.button
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={() => setIsAIAssistantOpen(true)}
          className="fixed bottom-6 right-24 flex items-center space-x-2 px-4 py-3 bg-gradient-to-r from-indigo-500 to-purple-600 text-white rounded-xl shadow-lg hover:shadow-xl transition-all duration-300"
          title="Open AI Assistant"
        >
          <Bot className="w-5 h-5" />
          <span className="font-medium">AI Assistant</span>
        </motion.button>

        {/* Save Modal */}
        <SaveEvaluationModal
          isOpen={isSaveModalOpen}
          onClose={() => setIsSaveModalOpen(false)}
          onSave={handleSaveEvaluation}
          evaluationData={dashboardData}
        />

        {/* AI Assistant */}
        <AIAssistant
          isOpen={isAIAssistantOpen}
          onClose={() => setIsAIAssistantOpen(false)}
          messages={aiMessages}
          onSendMessage={sendAIMessage}
          isTyping={isAITyping}
          sessionId={aiSessionId}
        />
      </div>
    );
  }

  if (currentView === 'history') {
    return (
      <EvaluationHistory
        onLoadEvaluation={handleLoadEvaluation}
        onBackToLanding={handleBackToLanding}
        onNewEvaluation={handleNewEvaluation}
        evaluationStats={evaluationStats}
      />
    );
  }

  if (currentView === 'ai-eval') {
    return (
      <AIEvaluationPanel
        onEvaluationComplete={handleAIEvaluation}
        onBackToLanding={handleBackToLanding}
        aiFeatures={aiFeatures}
        onAIFeaturesChange={setAIFeatures}
        evaluationMode={evaluationMode}
        onEvaluationModeChange={setEvaluationMode}
      />
    );
  }

  // Landing Page (Enhanced with AI Features)
  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-purple-50 relative overflow-hidden">
      {/* Enhanced Background decoration */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-1/2 -right-1/2 w-96 h-96 bg-gradient-to-br from-blue-400/20 to-purple-600/20 rounded-full blur-3xl"></div>
        <div className="absolute -bottom-1/2 -left-1/2 w-96 h-96 bg-gradient-to-tr from-green-400/20 to-blue-600/20 rounded-full blur-3xl"></div>
        <div className="absolute top-1/4 right-1/4 w-64 h-64 bg-gradient-to-bl from-purple-400/10 to-pink-600/10 rounded-full blur-2xl"></div>
      </div>

      {/* API Status Bar */}
      <div className="relative z-10 bg-white/80 backdrop-blur-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className={`flex items-center space-x-2 px-3 py-1 rounded-full text-sm font-medium ${
                apiHealth === 'healthy' ? 'bg-green-100 text-green-800' :
                apiHealth === 'unhealthy' ? 'bg-red-100 text-red-800' :
                'bg-yellow-100 text-yellow-800'
              }`}>
                <div className={`w-2 h-2 rounded-full ${
                  apiHealth === 'healthy' ? 'bg-green-600' :
                  apiHealth === 'unhealthy' ? 'bg-red-600 animate-pulse' :
                  'bg-yellow-600 animate-pulse'
                }`} />
                <span>
                  {apiHealth === 'healthy' ? 'Backend Connected' :
                   apiHealth === 'unhealthy' ? 'Backend Disconnected' :
                   'Checking Backend...'}
                </span>
              </div>

              {evaluationStats && (
                <div className="flex items-center space-x-2 px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm font-medium">
                  <History className="w-3 h-3" />
                  <span>{evaluationStats.total_evaluations} Evaluations Stored</span>
                </div>
              )}

              <div className="flex items-center space-x-2 px-3 py-1 bg-purple-100 text-purple-800 rounded-full text-sm font-medium">
                <Brain className="w-3 h-3" />
                <span>AI-Enhanced</span>
              </div>
            </div>
            
            <div className="flex items-center space-x-3 text-sm text-gray-600">
              <div className="flex items-center space-x-1">
                <Shield className="w-4 h-4 text-green-600" />
                <span>Secure</span>
              </div>
              <div className="flex items-center space-x-1">
                <Zap className="w-4 h-4 text-yellow-600" />
                <span>Fast</span>
              </div>
              <div className="flex items-center space-x-1">
                <Bot className="w-4 h-4 text-purple-600" />
                <span>AI-Powered</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Header Section */}
      <header className="relative z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
          <div className="text-center">
            {/* Badge */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="inline-flex items-center px-6 py-3 bg-white/90 backdrop-blur-sm rounded-full border border-indigo-200 text-indigo-700 text-sm font-semibold mb-8 shadow-lg"
            >
              <Bot className="w-5 h-5 mr-2" />
              Next-Generation AI-Powered Evaluation Platform
              <span className="ml-2 px-2 py-1 bg-indigo-100 rounded-full text-xs">v3.0</span>
            </motion.div>
            
            {/* Main Title */}
            <motion.h1 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="text-5xl md:text-6xl lg:text-5xl font-extrabold text-gray-900 mb-8 leading-tight"
            >
              <span className="bg-gradient-to-r from-indigo-600 via-purple-600 to-blue-600 bg-clip-text text-transparent">
                AI Agent
              </span>
              <span className="text-gray-800 pl-3">Evaluator</span>
              <div className="text-2xl md:text-3xl lg:text-4xl font-semibold text-purple-600 mt-4">
                <Sparkles className="inline w-8 h-8 mr-2" />
                With AI Assistant
              </div>
            </motion.h1>
            
            {/* Feature Highlights */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="flex flex-wrap items-center justify-center gap-4 mb-8"
            >
              <div className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-blue-500/10 to-purple-500/10 rounded-full">
                <Wand2 className="w-4 h-4 text-blue-600" />
                <span className="text-sm font-medium text-gray-700">Auto Reference Generation</span>
              </div>
              <div className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-green-500/10 to-blue-500/10 rounded-full">
                <Lightbulb className="w-4 h-4 text-green-600" />
                <span className="text-sm font-medium text-gray-700">Prompt Optimization</span>
              </div>
              <div className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-purple-500/10 to-pink-500/10 rounded-full">
                <MessageCircle className="w-4 h-4 text-purple-600" />
                <span className="text-sm font-medium text-gray-700">Interactive AI Support</span>
              </div>
            </motion.div>
            
            {/* CTA Buttons */}
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="flex flex-col sm:flex-row items-center justify-center space-y-4 sm:space-y-0 sm:space-x-6 mb-12"
            >
              <button
                onClick={handleAIEvaluationMode}
                disabled={isLoading || apiHealth === 'unhealthy'}
                className="group inline-flex items-center px-8 py-4 text-lg font-semibold rounded-2xl text-white bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transform hover:scale-105 transition-all duration-300 shadow-lg hover:shadow-2xl disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
              >
                <Bot className="w-6 h-6 mr-3 group-hover:animate-bounce" />
                {isLoading ? 'Processing...' : 
                 apiHealth === 'unhealthy' ? 'Backend Required' : 
                 'AI Evaluation'}
                <Sparkles className="w-5 h-5 ml-2 group-hover:rotate-12 transition-transform" />
              </button>
              
              <button
                onClick={handleNewEvaluation}
                disabled={isLoading || apiHealth === 'unhealthy'}
                className="group inline-flex items-center px-6 py-3 text-lg font-medium rounded-2xl text-gray-700 bg-white/80 backdrop-blur-sm border border-gray-200 hover:bg-white hover:border-gray-300 transition-all duration-300"
              >
                <Upload className="w-5 h-5 mr-2" />
                Upload Data
              </button>
              
              <button
                onClick={handleViewHistory}
                className="group inline-flex items-center px-6 py-3 text-lg font-medium rounded-2xl text-gray-700 bg-white/80 backdrop-blur-sm border border-gray-200 hover:bg-white hover:border-gray-300 transition-all duration-300"
              >
                <History className="w-5 h-5 mr-2" />
                View History
                {evaluationStats && evaluationStats.total_evaluations > 0 && (
                  <span className="ml-2 px-2 py-1 bg-purple-100 text-purple-700 text-xs rounded-full">
                    {evaluationStats.total_evaluations}
                  </span>
                )}
              </button>
            </motion.div>
          </div>
        </div>
      </header>

      {/* Enhanced Stats Section */}
      <section className="py-20 relative z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl font-bold text-gray-900 mb-4">
              Comprehensive AI-Powered Framework
            </h2>
            <p className="text-lg text-gray-600 max-w-3xl mx-auto">
              Our advanced platform provides deep insights into AI performance with persistent storage, 
              AI-generated explanations, and intelligent prompt optimization to help you achieve better results.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {stats.map((stat, index) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: stat.delay }}
                className="group relative"
              >
                <div className="absolute -inset-1 bg-gradient-to-r from-indigo-600 to-purple-600 rounded-2xl blur opacity-25 group-hover:opacity-40 transition duration-300"></div>
                <div className="relative bg-white/95 backdrop-blur-sm rounded-2xl p-8 flex flex-col items-center justify-between min-h-[320px] hover:bg-white transition-all duration-300 border border-gray-200 hover:border-indigo-300 shadow-lg hover:shadow-xl">
                  <div className={`inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-r ${stat.color} text-white mb-6 group-hover:scale-110 transition-transform duration-300 shadow-lg`}>
                    <stat.icon className="w-8 h-8" />
                  </div>
                  <div className="text-4xl font-bold text-gray-900 mb-2">
                    {stat.value === '∞' ? '∞' : (
                      <AnimatedCounter 
                        end={parseInt(stat.value) || 5} 
                        suffix={stat.suffix} 
                        duration={2}
                      />
                    )}
                  </div>
                  <div className="font-semibold text-gray-800 mb-3 text-center">
                    {stat.label}
                  </div>
                  <div className="text-sm text-gray-600 leading-relaxed text-center line-clamp-3">
                    {stat.description}
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* AI Features Showcase */}
      <section className="py-20 bg-gradient-to-r from-indigo-50 to-purple-50 relative z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl font-bold text-gray-900 mb-4">
              AI-Powered Evaluation Features
            </h2>
            <p className="text-lg text-gray-600 max-w-3xl mx-auto">
              Experience the future of AI evaluation with intelligent automation and actionable insights.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.1 }}
              className="bg-white rounded-2xl p-8 shadow-xl hover:shadow-2xl transition-all duration-300 border border-indigo-100"
            >
              <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-indigo-600 rounded-2xl flex items-center justify-center mb-6">
                <Brain className="w-8 h-8 text-white" />
              </div>
              <h3 className="text-2xl font-bold text-gray-900 mb-4">AI Judge System</h3>
              <p className="text-gray-600 mb-4">
                Advanced LLM-powered scoring across all 7 evaluation metrics with detailed explanations 
                for every score and transparent reasoning.
              </p>
              <div className="flex items-center text-sm text-indigo-600 font-medium">
                <Sparkles className="w-4 h-4 mr-2" />
                Powered by GPT-4
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="bg-white rounded-2xl p-8 shadow-xl hover:shadow-2xl transition-all duration-300 border border-purple-100"
            >
              <div className="w-16 h-16 bg-gradient-to-r from-purple-500 to-pink-600 rounded-2xl flex items-center justify-center mb-6">
                <Wand2 className="w-8 h-8 text-white" />
              </div>
              <h3 className="text-2xl font-bold text-gray-900 mb-4">Smart Automation</h3>
              <p className="text-gray-600 mb-4">
                Automatically generate reference answers, explanations, and improvement suggestions. 
                No manual annotation required for comprehensive evaluations.
              </p>
              <div className="flex items-center text-sm text-purple-600 font-medium">
                <Lightbulb className="w-4 h-4 mr-2" />
                Fully Automated
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 }}
              className="bg-white rounded-2xl p-8 shadow-xl hover:shadow-2xl transition-all duration-300 border border-green-100"
            >
              <div className="w-16 h-16 bg-gradient-to-r from-green-500 to-teal-600 rounded-2xl flex items-center justify-center mb-6">
                <MessageCircle className="w-8 h-8 text-white" />
              </div>
              <h3 className="text-2xl font-bold text-gray-900 mb-4">Interactive Support</h3>
              <p className="text-gray-600 mb-4">
                Chat with your AI assistant for instant help, metric explanations, and personalized 
                recommendations to improve your AI agents.
              </p>
              <div className="flex items-center text-sm text-green-600 font-medium">
                <Bot className="w-4 h-4 mr-2" />
                24/7 Available
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Enhanced Notification System */}
      <AnimatePresence>
        {notification && (
          <motion.div
            initial={{ opacity: 0, y: 50, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 50, scale: 0.9 }}
            className={`fixed bottom-6 right-6 p-6 rounded-2xl shadow-2xl z-50 max-w-md border backdrop-blur-sm ${
              notification.type === 'success' ? 'bg-green-50/95 border-green-200 text-green-800' :
              notification.type === 'error' ? 'bg-red-50/95 border-red-200 text-red-800' :
              notification.type === 'warning' ? 'bg-yellow-50/95 border-yellow-200 text-yellow-800' :
              'bg-blue-50/95 border-blue-200 text-blue-800'
            }`}
          >
            <div className="flex items-start space-x-3">
              <div className="flex-shrink-0 pt-1">
                {notification.type === 'success' && <CheckCircle className="w-6 h-6 text-green-600" />}
                {notification.type === 'error' && <AlertTriangle className="w-6 h-6 text-red-600" />}
                {notification.type === 'warning' && <AlertTriangle className="w-6 h-6 text-yellow-600" />}
                {notification.type === 'info' && <HelpCircle className="w-6 h-6 text-blue-600" />}
              </div>
              <div className="flex-1">
                <p className="font-medium mb-1">
                  {notification.type === 'success' ? 'Success!' :
                   notification.type === 'error' ? 'Error' :
                   notification.type === 'warning' ? 'Warning' :
                   'Information'}
                </p>
                <p className="text-sm opacity-90">{notification.message}</p>
              </div>
              <button
                onClick={() => setNotification(null)}
                className="flex-shrink-0 text-gray-400 hover:text-gray-600 transition-colors p-1 rounded-full hover:bg-white/50"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Enhanced Loading Overlay */}
      {isLoading && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50"
        >
          <div className="bg-white rounded-2xl p-8 flex flex-col items-center space-y-4 shadow-2xl border border-gray-200 max-w-sm mx-4">
            <div className="relative">
              <RefreshCw className="w-8 h-8 animate-spin text-indigo-600" />
              <div className="absolute inset-0 w-8 h-8 border-2 border-purple-200 border-t-purple-600 rounded-full animate-spin"></div>
            </div>
            <div className="text-center">
              <h3 className="text-lg font-semibold text-gray-900 mb-2 flex items-center">
                <Brain className="w-5 h-5 mr-2 text-purple-600" />
                AI Processing
              </h3>
              <p className="text-sm text-gray-600">
                {loadingMessage || 'Please wait while our AI analyzes your request...'}
              </p>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div className="bg-gradient-to-r from-indigo-500 to-purple-600 h-2 rounded-full animate-pulse"></div>
            </div>
          </div>
        </motion.div>
      )}

      {/* Upload Panel */}
      <UploadPanel
        isOpen={isUploadOpen}
        onClose={() => setIsUploadOpen(false)}
        onUploaded={handleUploadComplete}
        onError={handleUploadError}
        evaluationMode={evaluationMode}
        onEvaluationModeChange={setEvaluationMode}
        aiFeatures={aiFeatures}
        onAIFeaturesChange={setAIFeatures}
      />

      {/* AI Assistant (Global) */}
      <AIAssistant
        isOpen={isAIAssistantOpen}
        onClose={() => setIsAIAssistantOpen(false)}
        messages={aiMessages}
        onSendMessage={sendAIMessage}
        isTyping={isAITyping}
        sessionId={aiSessionId}
      />

      {/* Floating AI Assistant Button */}
      <motion.button
        initial={{ opacity: 0, scale: 0 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 1 }}
        onClick={() => setIsAIAssistantOpen(true)}
        className="fixed bottom-6 right-6 w-14 h-14 bg-gradient-to-r from-indigo-500 to-purple-600 text-white rounded-full shadow-lg hover:shadow-xl transition-all duration-300 flex items-center justify-center z-40 group"
        title="Open AI Assistant"
      >
        <Bot className="w-6 h-6 group-hover:scale-110 transition-transform" />
        <div className="absolute -top-2 -right-2 w-4 h-4 bg-green-500 rounded-full animate-pulse"></div>
      </motion.button>
    </div>
  );
};

export default App;