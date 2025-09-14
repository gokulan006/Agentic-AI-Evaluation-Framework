import React, { useState, useMemo, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Line, ScatterChart, Scatter, Cell,
  ComposedChart
} from 'recharts';
import { 
  Download,  Search, ChevronDown, TrendingUp, TrendingDown,
  Award, AlertTriangle, MessageCircle, FileText, BarChart3, Radar as RadarIcon,
  Grid3x3, List, Download as DownloadIcon, Star, Trophy, Target,
  Brain, Zap, Shield, CheckCircle, RefreshCw, Copy,
  Maximize2, Minimize2, SortAsc, SortDesc, Bookmark,Activity,
  X, Bot, Lightbulb, Wand2, Sparkles, 
} from 'lucide-react';

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
  metrics: Metrics;
  generated_at: string;
  explanation?: Record<string, string>;
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
}

interface ComprehensiveDashboardProps {
  data: DashboardData;
  onExport?: (type: 'pdf' | 'excel' | 'csv') => Promise<void>;
  onGetExplanation?: (result: EvaluationResult, metric: string) => Promise<string>;
  evaluationId?: string | null;
  onSave?: () => void;
  aiFeatures?: any;
  evaluationMode?: string;
}

// New interface for AI suggestions
interface AISuggestion {
  explanation: string;
  suggestedPrompt: string;
  analysis: string;
  expectedImprovements: string[];
  targetMetrics: string[];
}

// Enhanced Constants and Utilities
const metricLabels: Record<keyof Metrics, string> = {
  instruction_score: 'Instruction Following',
  hallucination_score: 'Hallucination Detection',
  assumption_score: 'Assumption Handling', 
  coherence_score: 'Coherence',
  accuracy_score: 'Accuracy',
  completeness_score: 'Completeness',
  overall_score: 'Overall Quality'
};

const metricIcons: Record<keyof Metrics, React.ElementType> = {
  instruction_score: Target,
  hallucination_score: Shield,
  assumption_score: Brain,
  coherence_score: Zap,
  accuracy_score: CheckCircle,
  completeness_score: Star,
  overall_score: Trophy
};

 

const colors = [
  '#8884d8', '#82ca9d', '#ffc658', '#ff7c7c', '#8dd1e1', '#d084d0', '#ffb366',
  '#a4de6c', '#ffc0cb', '#40e0d0', '#da70d6', '#32cd32', '#ff6347'
];

// Utility Functions
const getScoreColor = (score: number) => {
  if (score >= 0.8) return 'text-green-600 bg-green-50 border-green-200';
  if (score >= 0.6) return 'text-yellow-600 bg-yellow-50 border-yellow-200';
  return 'text-red-600 bg-red-50 border-red-200';
};

const getScoreColorHex = (score: number) => {
  if (score >= 0.8) return '#10b981';
  if (score >= 0.6) return '#f59e0b';
  return '#ef4444';
};

const getPerformanceGrade = (score: number) => {
  if (score >= 0.9) return { grade: 'A+', color: 'text-green-700', bg: 'bg-green-100' };
  if (score >= 0.8) return { grade: 'A', color: 'text-green-600', bg: 'bg-green-100' };
  if (score >= 0.7) return { grade: 'B+', color: 'text-blue-600', bg: 'bg-blue-100' };
  if (score >= 0.6) return { grade: 'B', color: 'text-blue-500', bg: 'bg-blue-100' };
  if (score >= 0.5) return { grade: 'C', color: 'text-yellow-600', bg: 'bg-yellow-100' };
  return { grade: 'D', color: 'text-red-600', bg: 'bg-red-100' };
};

const calculateVariance = (scores: number[]) => {
  const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
  return scores.reduce((acc, score) => acc + Math.pow(score - mean, 2), 0) / scores.length;
};

const findOutliers = (scores: number[]) => {
  const sorted = [...scores].sort((a, b) => a - b);
  const q1 = sorted[Math.floor(sorted.length * 0.25)];
  const q3 = sorted[Math.floor(sorted.length * 0.75)];
  const iqr = q3 - q1;
  const lowerBound = q1 - 1.5 * iqr;
  const upperBound = q3 + 1.5 * iqr;
  return scores.filter(score => score < lowerBound || score > upperBound);
};

// Animated Counter Component
const AnimatedCounter: React.FC<{
  end: number;
  duration?: number;
  suffix?: string;
  prefix?: string;
  decimals?: number;
  className?: string;
}> = ({ end, duration = 2, suffix = '', prefix = '', decimals = 0, className = '' }) => {
  const [count, setCount] = useState(0);
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !isVisible) {
          setIsVisible(true);
          let startTime: number;
          
          const animate = (timestamp: number) => {
            if (!startTime) startTime = timestamp;
            const progress = Math.min((timestamp - startTime) / (duration * 1000), 1);
            const easeOutCubic = 1 - Math.pow(1 - progress, 3);
            setCount(end * easeOutCubic);
            
            if (progress < 1) {
              requestAnimationFrame(animate);
            }
          };
          
          requestAnimationFrame(animate);
        }
      },
      { threshold: 0.1 }
    );

    const element = document.getElementById(`counter-${end}-${suffix}`);
    if (element) observer.observe(element);

    return () => observer.disconnect();
  }, [end, duration, suffix, isVisible]);

  return (
    <span id={`counter-${end}-${suffix}`} className={className}>
      {prefix}{count.toFixed(decimals)}{suffix}
    </span>
  );
};

// AI Suggestion Modal Component
const AISuggestionModal: React.FC<{
  isOpen: boolean;
  onClose: () => void;
  result: EvaluationResult | null;
  suggestion: AISuggestion | null;
  isLoading: boolean;
}> = ({ isOpen, onClose, result, suggestion, isLoading }) => {
  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4"
        onClick={(e) => e.target === e.currentTarget && onClose()}
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.9, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.9, y: 20 }}
          className="bg-white rounded-2xl shadow-2xl w-full max-w-4xl max-h-[90vh] overflow-hidden"
        >
          {/* Header */}
          <div className="bg-gradient-to-r from-indigo-500 to-purple-600 text-white p-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-white/20 rounded-lg">
                  <Bot className="w-6 h-6" />
                </div>
                <div>
                  <h3 className="text-xl font-bold">AI Analysis & Prompt Optimization</h3>
                  <p className="text-indigo-100 text-sm">
                    {result ? `Analysis for: ${result.prompt_id} - ${result.agent_id}` : 'Loading...'}
                  </p>
                </div>
              </div>
              <button
                onClick={onClose}
                className="p-2 hover:bg-white/20 rounded-lg transition-colors"
              >
                <X className="w-6 h-6" />
              </button>
            </div>
          </div>

          {/* Content */}
          <div className="p-6 overflow-y-auto max-h-[calc(90vh-140px)]">
            {isLoading ? (
              <div className="flex flex-col items-center justify-center py-12">
                <RefreshCw className="w-8 h-8 animate-spin text-indigo-600 mb-4" />
                <h4 className="text-lg font-semibold text-gray-900 mb-2">AI is analyzing...</h4>
                <p className="text-gray-600 text-center max-w-md">
                  Our AI is reviewing the prompt, response, and scores to provide detailed insights and improvement suggestions.
                </p>
              </div>
            ) : suggestion && result ? (
              <div className="space-y-6">
                {/* Current Evaluation Overview */}
                <div className="bg-gray-50 rounded-xl p-6">
                  <h4 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                    <FileText className="w-5 h-5 mr-2 text-indigo-600" />
                    Current Evaluation
                  </h4>
                  <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">Original Prompt</label>
                      <div className="bg-white p-3 rounded-lg border text-sm text-gray-800 max-h-24 overflow-y-auto">
                        {result.prompt}
                      </div>
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">Agent Response</label>
                      <div className="bg-white p-3 rounded-lg border text-sm text-gray-800 max-h-24 overflow-y-auto">
                        {result.response}
                      </div>
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">Overall Score</label>
                      <div className={`inline-flex items-center px-4 py-2 rounded-lg font-semibold ${getScoreColor(result.metrics.overall_score)}`}>
                        <Trophy className="w-4 h-4 mr-2" />
                        {(result.metrics.overall_score * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>
                </div>

                {/* AI Explanation */}
                <div className="bg-blue-50 rounded-xl p-6 border border-blue-200">
                  <h4 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                    <Brain className="w-5 h-5 mr-2 text-blue-600" />
                    AI Explanation & Analysis
                  </h4>
                  <div className="bg-white p-4 rounded-lg border">
                    <p className="text-gray-800 leading-relaxed whitespace-pre-wrap">{suggestion.explanation}</p>
                  </div>
                  <div className="mt-4 flex justify-end">
                    <button
                      onClick={() => copyToClipboard(suggestion.explanation)}
                      className="flex items-center space-x-2 px-3 py-2 bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 transition-colors text-sm"
                    >
                      <Copy className="w-4 h-4" />
                      <span>Copy Explanation</span>
                    </button>
                  </div>
                </div>

                {/* Improvement Analysis */}
                <div className="bg-yellow-50 rounded-xl p-6 border border-yellow-200">
                  <h4 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                    <Lightbulb className="w-5 h-5 mr-2 text-yellow-600" />
                    Improvement Analysis
                  </h4>
                  <div className="bg-white p-4 rounded-lg border mb-4">
                    <p className="text-gray-800 leading-relaxed">{suggestion.analysis}</p>
                  </div>
                  
                  {suggestion.targetMetrics.length > 0 && (
                    <div>
                      <h5 className="font-medium text-gray-900 mb-2">Target Metrics for Improvement:</h5>
                      <div className="flex flex-wrap gap-2">
                        {suggestion.targetMetrics.map((metric, index) => (
                          <span key={index} className="px-3 py-1 bg-yellow-100 text-yellow-800 text-sm rounded-full">
                            {metricLabels[metric as keyof Metrics] || metric}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {/* Suggested Prompt */}
                <div className="bg-green-50 rounded-xl p-6 border border-green-200">
                  <h4 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                    <Wand2 className="w-5 h-5 mr-2 text-green-600" />
                    Optimized Prompt Suggestion
                  </h4>
                  <div className="bg-white p-4 rounded-lg border mb-4">
                    <p className="text-gray-800 leading-relaxed font-mono text-sm">{suggestion.suggestedPrompt}</p>
                  </div>
                  <div className="flex items-center justify-between">
                    <div>
                      <h5 className="font-medium text-gray-900 mb-2">Expected Improvements:</h5>
                      <ul className="text-sm text-gray-700 space-y-1">
                        {suggestion.expectedImprovements.map((improvement, index) => (
                          <li key={index} className="flex items-start space-x-2">
                            <CheckCircle className="w-4 h-4 text-green-600 mt-0.5 flex-shrink-0" />
                            <span>{improvement}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                    <button
                      onClick={() => copyToClipboard(suggestion.suggestedPrompt)}
                      className="flex items-center space-x-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
                    >
                      <Copy className="w-4 h-4" />
                      <span>Copy Prompt</span>
                    </button>
                  </div>
                </div>

                {/* Action Buttons */}
                <div className="flex justify-between items-center pt-4 border-t border-gray-200">
                  <div className="flex items-center space-x-2 text-sm text-gray-600">
                    <Sparkles className="w-4 h-4" />
                    <span>Analysis powered by AI • Generated at {new Date().toLocaleTimeString()}</span>
                  </div>
                  <div className="flex space-x-3">
                    <button
                      onClick={() => {
                        const fullAnalysis = `
AI Analysis for ${result.prompt_id} - ${result.agent_id}

Original Prompt:
${result.prompt}

Agent Response:
${result.response}

Overall Score: ${(result.metrics.overall_score * 100).toFixed(1)}%

AI Explanation:
${suggestion.explanation}

Improvement Analysis:
${suggestion.analysis}

Suggested Prompt:
${suggestion.suggestedPrompt}

Expected Improvements:
${suggestion.expectedImprovements.map(imp => `• ${imp}`).join('\n')}
                        `.trim();
                        copyToClipboard(fullAnalysis);
                      }}
                      className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
                    >
                      Copy Full Analysis
                    </button>
                    <button
                      onClick={onClose}
                      className="px-6 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
                    >
                      Close
                    </button>
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center py-12">
                <AlertTriangle className="w-8 h-8 text-red-500 mb-4" />
                <h4 className="text-lg font-semibold text-gray-900 mb-2">Analysis Failed</h4>
                <p className="text-gray-600 text-center max-w-md">
                  Unable to generate AI analysis. Please try again or check your API configuration.
                </p>
              </div>
            )}
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

// Enhanced Summary Cards Component
const SummaryCards: React.FC<{ data: DashboardData }> = ({ data }) => {
  const insights = useMemo(() => {
    const agents = Object.entries(data.agent_summaries);
    const bestAgent = agents.reduce((best, [agentId, metrics]) => 
      metrics.overall_score > best.score ? { id: agentId, score: metrics.overall_score } : best,
      { id: '', score: 0 }
    );

    const worstAgent = agents.reduce((worst, [agentId, metrics]) => 
      metrics.overall_score < worst.score ? { id: agentId, score: metrics.overall_score } : worst,
      { id: '', score: 1 }
    );

    const overallScores = data.results.map(r => r.metrics.overall_score);
    const variance = calculateVariance(overallScores);
    const outliers = findOutliers(overallScores);

    return {
      bestAgent,
      worstAgent,
      variance,
      outlierCount: outliers.length,
      avgScore: data.overall_summary.avg_overall_score,
      grade: getPerformanceGrade(data.overall_summary.avg_overall_score)
    };
  }, [data]);

  const stats = [
    {
      label: 'Total Evaluations',
      value: data.total_evaluated,
      icon: FileText,
      color: 'from-blue-500 to-blue-600',
      description: 'Comprehensive evaluations completed',
      trend: '+12%'
    },
    {
      label: 'Agents Analyzed',
      value: data.overall_summary.total_agents,
      icon: Brain,
      color: 'from-purple-500 to-purple-600',
      description: 'AI agents under evaluation',
      trend: null
    },
    {
      label: 'Performance Score',
      value: `${(data.overall_summary.avg_overall_score * 100).toFixed(1)}%`,
      icon: Target,
      color: 'from-green-500 to-green-600',
      description: `Grade: ${insights.grade.grade}`,
      trend: insights.avgScore > 0.7 ? '+5.2%' : '-2.1%'
    },
    {
      label: 'Best Performer',
      value: insights.bestAgent.id,
      subValue: `${(insights.bestAgent.score * 100).toFixed(1)}%`,
      icon: Trophy,
      color: 'from-yellow-500 to-yellow-600',
      description: 'Top scoring agent',
      trend: '+8.7%'
    }
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
      {stats.map((stat, index) => (
        <motion.div
          key={stat.label}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.1 }}
          className="bg-white rounded-xl shadow-lg overflow-hidden hover:shadow-xl transition-shadow duration-300"
        >
          <div className={`h-2 bg-gradient-to-r ${stat.color}`} />
          <div className="p-6">
            <div className="flex items-center justify-between mb-4">
              <div className={`p-3 rounded-xl bg-gradient-to-r ${stat.color} shadow-lg`}>
                <stat.icon className="w-6 h-6 text-white" />
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold text-gray-900">
                  {typeof stat.value === 'number' ? (
                    <AnimatedCounter 
                      end={stat.value}
                      duration={1.5}
                    />
                  ) : (
                    stat.value
                  )}
                </div>
                {stat.subValue && (
                  <div className="text-sm text-gray-600 font-medium">{stat.subValue}</div>
                )}
              </div>
            </div>
            
            <div className="space-y-2">
              <h3 className="text-sm font-semibold text-gray-800">{stat.label}</h3>
              <p className="text-xs text-gray-600">{stat.description}</p>
              {stat.trend && (
                <div className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                  stat.trend.startsWith('+') ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                }`}>
                  {stat.trend.startsWith('+') ? <TrendingUp className="w-3 h-3 mr-1" /> : <TrendingDown className="w-3 h-3 mr-1" />}
                  {stat.trend}
                </div>
              )}
            </div>
          </div>
        </motion.div>
      ))}
    </div>
  );
};

// Enhanced Radar Chart Component
const AgentRadarChart: React.FC<{ data: DashboardData }> = ({ data }) => {
  const [selectedAgents, setSelectedAgents] = useState<string[]>([]);
  const [showComparison, setShowComparison] = useState(false);
  
  const radarData = useMemo(() => {
    const metrics = Object.keys(metricLabels).filter(key => key !== 'overall_score') as (keyof Metrics)[];
    return metrics.map(metric => {
      const item: any = { 
        metric: metricLabels[metric],
        fullName: metric
      };
      Object.entries(data.agent_summaries).forEach(([agentId, agentMetrics]) => {
        item[agentId] = agentMetrics[metric];
      });
      return item;
    });
  }, [data]);

  const agents = Object.keys(data.agent_summaries);
  const visibleAgents = selectedAgents.length === 0 ? agents : selectedAgents;

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between mb-6 gap-4">
        <div className="flex items-center space-x-3">
          <RadarIcon className="w-6 h-6 text-indigo-600" />
          <div>
            <h3 className="text-xl font-semibold text-gray-900">Agent Performance Radar</h3>
            <p className="text-sm text-gray-600">Multi-dimensional performance comparison</p>
          </div>
        </div>
        
        <div className="flex flex-col sm:flex-row gap-3">
          <button
            onClick={() => setShowComparison(!showComparison)}
            className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors whitespace-nowrap ${
              showComparison ? 'bg-indigo-100 text-indigo-800' : 'bg-gray-100 text-gray-600'
            }`}
          >
            Comparison Mode
          </button>
          
          <div className="flex flex-wrap gap-2 max-w-md">
            {agents.map((agent, index) => (
              <button
                key={agent}
                onClick={() => setSelectedAgents(prev => 
                  prev.includes(agent) 
                    ? prev.filter(a => a !== agent)
                    : [...prev, agent]
                )}
                className={`px-3 py-1 rounded-full text-xs font-medium transition-all border flex-shrink-0 ${
                  selectedAgents.length === 0 || selectedAgents.includes(agent)
                    ? 'border-current'
                    : 'border-gray-300 text-gray-500'
                }`}
                style={{ 
                  backgroundColor: selectedAgents.length === 0 || selectedAgents.includes(agent) 
                    ? colors[index % colors.length] + '20' 
                    : 'white',
                  color: selectedAgents.length === 0 || selectedAgents.includes(agent)
                    ? colors[index % colors.length]
                    : undefined
                }}
              >
                {agent}
              </button>
            ))}
          </div>
        </div>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <ResponsiveContainer width="100%" height={400}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="#e5e7eb" />
              <PolarAngleAxis 
                dataKey="metric" 
                tick={{ fontSize: 12, fill: '#6b7280' }}
                className="text-xs"
              />
              <PolarRadiusAxis 
                domain={[0, 1]} 
                tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                tick={{ fontSize: 10, fill: '#9ca3af' }}
              />
              {visibleAgents.map((agent, index) => (
                <Radar
                  key={agent}
                  name={agent}
                  dataKey={agent}
                  stroke={colors[index % colors.length]}
                  fill={colors[index % colors.length]}
                  fillOpacity={0.15}
                  strokeWidth={3}
                  dot={{ fill: colors[index % colors.length], strokeWidth: 2, r: 4 }}
                />
              ))}
              <Tooltip 
                formatter={(value: number, name: string) => [`${(value * 100).toFixed(1)}%`, name]}
                labelFormatter={(label) => `Metric: ${label}`}
                contentStyle={{
                  backgroundColor: 'rgba(255, 255, 255, 0.95)',
                  border: '1px solid #e5e7eb',
                  borderRadius: '8px',
                  boxShadow: '0 10px 25px rgba(0, 0, 0, 0.1)'
                }}
              />
              <Legend 
                verticalAlign="bottom" 
                height={36}
                wrapperStyle={{ paddingTop: '20px' }}
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>
        
        {showComparison && (
          <div className="space-y-4">
            <h4 className="font-semibold text-gray-900">Performance Insights</h4>
            {Object.entries(metricLabels).filter(([key]) => key !== 'overall_score').map(([key, label]) => {
              const scores = visibleAgents.map(agent => data.agent_summaries[agent][key as keyof Metrics]);
              const best = Math.max(...scores);
              const worst = Math.min(...scores);
              const variance = calculateVariance(scores);
              
              return (
                <div key={key} className="bg-gray-50 rounded-lg p-3">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-gray-700">{label}</span>
                    <div className="flex flex-col sm:flex-row sm:items-center sm:space-x-2 gap-1 sm:gap-0">
                      <span className="text-xs text-green-600 bg-green-100 px-2 py-1 rounded">
                        Best: {(best * 100).toFixed(1)}%
                      </span>
                      <span className="text-xs text-red-600 bg-red-100 px-2 py-1 rounded">
                        Worst: {(worst * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                  <div className="text-xs text-gray-600">
                    Variance: {(variance * 10000).toFixed(1)} 
                    {variance > 0.01 && <span className="text-orange-600 ml-1">(High variation)</span>}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};

// Enhanced Metric Comparison Component
const MetricBarChart: React.FC<{ data: DashboardData }> = ({ data }) => {
  const [selectedMetric, setSelectedMetric] = useState<keyof Metrics>('overall_score');
  const [viewMode, setViewMode] = useState<'grouped' | 'stacked' | 'normalized'>('grouped');
  const [sortBy, setSortBy] = useState<'name' | 'score'>('score');

  const barData = useMemo(() => {
    let processedData;
    
    if (viewMode === 'grouped') {
      processedData = Object.entries(data.agent_summaries).map(([agentId, metrics]) => ({
        agent: agentId,
        [selectedMetric]: metrics[selectedMetric],
        rank: 0
      }));
    } else if (viewMode === 'stacked') {
      processedData = Object.entries(data.agent_summaries).map(([agentId, metrics]) => ({
        agent: agentId,
        ...Object.fromEntries(
          Object.entries(metrics).filter(([key]) => key !== 'overall_score')
        )
      }));
    } else {
      // normalized view - show all metrics normalized to 0-1 scale
      processedData = Object.entries(data.agent_summaries).map(([agentId, metrics]) => {
        const normalizedMetrics: any = { agent: agentId };
        Object.entries(metrics).forEach(([key, value]) => {
          if (key !== 'overall_score') {
            normalizedMetrics[key] = value;
          }
        });
        return normalizedMetrics;
      });
    }

    // Sort data
    if (sortBy === 'score' && viewMode === 'grouped') {
      processedData.sort((a, b) => b[selectedMetric] - a[selectedMetric]);
      processedData.forEach((item, index) => {
        item.rank = index + 1;
      });
    } else if (sortBy === 'score' && viewMode !== 'grouped') {
      processedData.sort((a, b) => {
        const avgA = Object.values(a).filter(v => typeof v === 'number').reduce((sum: number, val) => sum + (val as number), 0) / 6;
        const avgB = Object.values(b).filter(v => typeof v === 'number').reduce((sum: number, val) => sum + (val as number), 0) / 6;
        return avgB - avgA;
      });
    } else {
      processedData.sort((a, b) => a.agent.localeCompare(b.agent));
    }

    return processedData;
  }, [data, selectedMetric, viewMode, sortBy]);

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
          <p className="font-semibold text-gray-900">{`Agent: ${label}`}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} style={{ color: entry.color }} className="text-sm">
              {`${metricLabels[entry.dataKey as keyof Metrics] || entry.dataKey}: ${(entry.value * 100).toFixed(1)}%`}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between mb-6 gap-4">
        <div className="flex items-center space-x-3">
          <BarChart3 className="w-6 h-6 text-indigo-600" />
          <div>
            <h3 className="text-xl font-semibold text-gray-900">Metric Comparison</h3>
            <p className="text-sm text-gray-600">Detailed performance analysis across agents</p>
          </div>
        </div>
        
        <div className="flex flex-col sm:flex-row gap-3 flex-wrap">
          <select
            value={viewMode}
            onChange={(e) => setViewMode(e.target.value as 'grouped' | 'stacked' | 'normalized')}
            className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 min-w-[140px]"
          >
            <option value="grouped">Single Metric</option>
            <option value="stacked">All Metrics</option>
            <option value="normalized">Normalized View</option>
          </select>
          
          {viewMode === 'grouped' && (
            <select
              value={selectedMetric}
              onChange={(e) => setSelectedMetric(e.target.value as keyof Metrics)}
              className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 min-w-[160px]"
            >
              {Object.entries(metricLabels).map(([key, label]) => (
                <option key={key} value={key}>{label}</option>
              ))}
            </select>
          )}
          
          <button
            onClick={() => setSortBy(sortBy === 'name' ? 'score' : 'name')}
            className="flex items-center px-3 py-2 border border-gray-300 rounded-lg text-sm hover:bg-gray-50 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 whitespace-nowrap"
          >
            {sortBy === 'score' ? <SortDesc className="w-4 h-4 mr-1" /> : <SortAsc className="w-4 h-4 mr-1" />}
            Sort by {sortBy === 'name' ? 'Name' : 'Score'}
          </button>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={400}>
        <BarChart data={barData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis 
            dataKey="agent" 
            tick={{ fontSize: 12 }}
            angle={-45}
            textAnchor="end"
            height={80}
          />
          <YAxis 
            tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
            tick={{ fontSize: 12 }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend />
          
          {viewMode === 'grouped' ? (
            <Bar 
              dataKey={selectedMetric} 
              name={metricLabels[selectedMetric]}
              radius={[4, 4, 0, 0]}
            >
              {barData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={getScoreColorHex(entry[selectedMetric as keyof typeof entry])} />
              ))}
            </Bar>
          ) : (
            Object.keys(metricLabels).filter(key => key !== 'overall_score').map((metric, index) => (
              <Bar 
                key={metric} 
                dataKey={metric} 
                fill={colors[index % colors.length]} 
                name={metricLabels[metric as keyof Metrics]}
                radius={index === 0 ? [4, 4, 0, 0] : [0, 0, 0, 0]}
              />
            ))
          )}
        </BarChart>
      </ResponsiveContainer>
      
      {viewMode === 'grouped' && (
        <div className="mt-4 flex flex-wrap gap-2">
          {barData.slice(0, 3).map((agent, index) => (
            <div key={agent.agent} className="flex items-center space-x-2 bg-gray-50 rounded-lg px-3 py-2">
              <div className={`w-3 h-3 rounded-full ${index === 0 ? 'bg-yellow-400' : index === 1 ? 'bg-gray-400' : 'bg-orange-400'}`} />
              <span className="text-sm font-medium">{agent.agent}</span>
              <span className="text-sm text-gray-600">{(agent[selectedMetric] * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

const MetricHeatmap: React.FC<{ data: DashboardData }> = ({ data }) => {
  const [selectedView, setSelectedView] = useState<'absolute' | 'relative'>('absolute');
  const [highlightOutliers, setHighlightOutliers] = useState(false);

  const agents = Object.keys(data.agent_summaries);
  const metrics = Object.keys(metricLabels) as (keyof Metrics)[];

  const heatmapData = useMemo(() => {
    const processedData: any[][] = [];
    
    agents.forEach((agent, agentIndex) => {
      const row: any[] = [];
      metrics.forEach((metric, metricIndex) => {
        const score = data.agent_summaries[agent][metric];
        let processedScore = score;
        
        if (selectedView === 'relative') {
          // Calculate relative performance (z-score)
          const allScoresForMetric = agents.map(a => data.agent_summaries[a][metric]);
          const mean = allScoresForMetric.reduce((a, b) => a + b, 0) / allScoresForMetric.length;
          const stdDev = Math.sqrt(allScoresForMetric.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / allScoresForMetric.length);
          processedScore = stdDev === 0 ? 0 : (score - mean) / stdDev;
        }
        
        // Check if it's an outlier
        const allScoresForMetric = agents.map(a => data.agent_summaries[a][metric]);
        const outliers = findOutliers(allScoresForMetric);
        const isOutlier = outliers.includes(score);
        
        row.push({
          agent: agentIndex,
          metric: metricIndex,
          score: processedScore,
          originalScore: score,
          isOutlier,
          agentName: agent,
          metricName: metricLabels[metric]
        });
      });
      processedData.push(row);
    });
    
    return processedData;
  }, [data, agents, metrics, selectedView]);

  const getHeatmapColor = (score: number, isRelative: boolean = false) => {
    if (isRelative) {
      // For z-scores, use a different color scheme
      const intensity = Math.max(0, Math.min(1, (score + 3) / 6)); // Normalize z-score to 0-1
      return `hsl(${240 - intensity * 120}, 70%, ${85 - intensity * 30}%)`;
    } else {
      // Use red-yellow-green gradient for absolute scores
      const hue = score * 120; // 0 (red) to 120 (green)
      const saturation = 70;
      const lightness = 85 - score * 30; // Darker for higher scores
      return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <Grid3x3 className="w-6 h-6 text-indigo-600" />
          <div>
            <h3 className="text-xl font-semibold text-gray-900">Performance Heatmap</h3>
            <p className="text-sm text-gray-600">Visual performance matrix across all metrics</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-4">
          <button
            onClick={() => setHighlightOutliers(!highlightOutliers)}
            className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
              highlightOutliers ? 'bg-red-100 text-red-800' : 'bg-gray-100 text-gray-600'
            }`}
          >
            {highlightOutliers ? 'Hide' : 'Show'} Outliers
          </button>
          
          <select
            value={selectedView}
            onChange={(e) => setSelectedView(e.target.value as 'absolute' | 'relative')}
            className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
          >
            <option value="absolute">Absolute Scores</option>
            <option value="relative">Relative Performance</option>
          </select>
        </div>
      </div>

      <div className="overflow-x-auto">
        <div className="min-w-full">
          <table className="w-full border-collapse">
            <thead>
              <tr>
                <th className="p-3 text-left font-semibold text-gray-900 border-b-2 border-gray-200 bg-gray-50">
                  Agent
                </th>
                {metrics.map(metric => (
                  <th key={metric} className="p-3 text-center text-xs font-semibold text-gray-700 border-b-2 border-gray-200 bg-gray-50 min-w-[100px]">
                    <div className="flex flex-col items-center space-y-1">
                      <div className="flex items-center space-x-1">
                        {React.createElement(metricIcons[metric], { className: "w-4 h-4" })}
                      </div>
                      <div className="leading-tight">
                        {metricLabels[metric].split(' ').map((word, i) => (
                          <div key={i}>{word}</div>
                        ))}
                      </div>
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {heatmapData.map((agentRow, agentIndex) => (
                <tr key={agents[agentIndex]} className="hover:bg-gray-50 transition-colors">
                  <td className="p-3 font-medium text-gray-900 border-b border-gray-100 bg-gray-50">
                    <div className="flex items-center space-x-2">
                      <div className="w-8 h-8 rounded-full bg-indigo-100 flex items-center justify-center text-sm font-semibold text-indigo-600">
                        {agents[agentIndex].charAt(0).toUpperCase()}
                      </div>
                      <span>{agents[agentIndex]}</span>
                    </div>
                  </td>
                  {agentRow.map((cell, metricIndex) => (
                    <td 
                      key={metricIndex}
                      className="p-3 text-center relative group border-b border-gray-100 transition-all duration-200"
                      style={{ 
                        backgroundColor: getHeatmapColor(cell.score, selectedView === 'relative'),
                        border: highlightOutliers && cell.isOutlier ? '2px solid #ef4444' : undefined
                      }}
                    >
                      <div className="font-semibold text-gray-900 relative z-10">
                        {selectedView === 'absolute' 
                          ? `${(cell.originalScore * 100).toFixed(0)}%`
                          : `${cell.score > 0 ? '+' : ''}${cell.score.toFixed(2)}σ`
                        }
                      </div>
                      
                      {highlightOutliers && cell.isOutlier && (
                        <div className="absolute top-1 right-1">
                          <AlertTriangle className="w-3 h-3 text-red-600" />
                        </div>
                      )}
                      
                      {/* Enhanced Tooltip */}
                      <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 bg-gray-900 text-white px-3 py-2 rounded-lg text-xs opacity-0 group-hover:opacity-100 transition-opacity duration-200 z-50 whitespace-nowrap shadow-lg">
                        <div className="font-semibold">{cell.agentName}</div>
                        <div>{cell.metricName}</div>
                        <div>Score: {(cell.originalScore * 100).toFixed(1)}%</div>
                        {selectedView === 'relative' && (
                          <div>Z-score: {cell.score.toFixed(2)}σ</div>
                        )}
                        {cell.isOutlier && <div className="text-red-300">Outlier detected</div>}
                      </div>
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      
      {/* Color Legend */}
      <div className="mt-6 flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <span className="text-sm font-medium text-gray-700">
            {selectedView === 'absolute' ? 'Performance Scale:' : 'Relative Performance Scale:'}
          </span>
          <div className="flex items-center space-x-2">
            {selectedView === 'absolute' ? (
              <>
                <div className="flex items-center space-x-1">
                  <div className="w-4 h-4 rounded" style={{ backgroundColor: getHeatmapColor(0) }} />
                  <span className="text-xs">Low</span>
                </div>
                <div className="flex items-center space-x-1">
                  <div className="w-4 h-4 rounded" style={{ backgroundColor: getHeatmapColor(0.5) }} />
                  <span className="text-xs">Medium</span>
                </div>
                <div className="flex items-center space-x-1">
                  <div className="w-4 h-4 rounded" style={{ backgroundColor: getHeatmapColor(1) }} />
                  <span className="text-xs">High</span>
                </div>
              </>
            ) : (
              <>
                <div className="flex items-center space-x-1">
                  <div className="w-4 h-4 rounded" style={{ backgroundColor: getHeatmapColor(-2, true) }} />
                  <span className="text-xs">Below Avg</span>
                </div>
                <div className="flex items-center space-x-1">
                  <div className="w-4 h-4 rounded" style={{ backgroundColor: getHeatmapColor(0, true) }} />
                  <span className="text-xs">Average</span>
                </div>
                <div className="flex items-center space-x-1">
                  <div className="w-4 h-4 rounded" style={{ backgroundColor: getHeatmapColor(2, true) }} />
                  <span className="text-xs">Above Avg</span>
                </div>
              </>
            )}
          </div>
        </div>
        
        {highlightOutliers && (
          <div className="flex items-center space-x-2 text-sm text-red-600">
            <AlertTriangle className="w-4 h-4" />
            <span>Outliers highlighted</span>
          </div>
        )}
      </div>
    </div>
  );
};
// Enhanced Detailed Results Table Component with AI Assistant
const DetailedResultsTable: React.FC<{ 
  data: DashboardData; 
  onGetExplanation?: (result: EvaluationResult, metric: string) => Promise<string>;
}> = ({ data, onGetExplanation }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedAgent, setSelectedAgent] = useState<string>('all');
  const [selectedMetric, setSelectedMetric] = useState<keyof Metrics>('overall_score');
  const [sortBy, setSortBy] = useState<keyof Metrics>('overall_score');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [expandedRows, setExpandedRows] = useState<Set<string>>(new Set());
  const [explanations, setExplanations] = useState<Record<string, string>>({});
  const [loadingExplanations, setLoadingExplanations] = useState<Set<string>>(new Set());
  const [showOnlyOutliers, setShowOnlyOutliers] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage] = useState(10);

  // New state for AI suggestions
  const [showAISuggestionModal, setShowAISuggestionModal] = useState(false);
  const [currentSuggestionResult, setCurrentSuggestionResult] = useState<EvaluationResult | null>(null);
  const [currentAISuggestion, setCurrentAISuggestion] = useState<AISuggestion | null>(null);
  const [isLoadingAISuggestion, setIsLoadingAISuggestion] = useState(false);

  const filteredResults = useMemo(() => {
    let results = data.results.filter(result => {
      const matchesSearch = result.prompt.toLowerCase().includes(searchTerm.toLowerCase()) ||
                          result.response.toLowerCase().includes(searchTerm.toLowerCase()) ||
                          result.prompt_id.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesAgent = selectedAgent === 'all' || result.agent_id === selectedAgent;
      
      if (showOnlyOutliers) {
        const scores = data.results
          .filter(r => r.agent_id === result.agent_id)
          .map(r => r.metrics[selectedMetric]);
        const outliers = findOutliers(scores);
        const isOutlier = outliers.includes(result.metrics[selectedMetric]);
        return matchesSearch && matchesAgent && isOutlier;
      }
      
      return matchesSearch && matchesAgent;
    });

    results.sort((a, b) => {
      const valueA = a.metrics[sortBy];
      const valueB = b.metrics[sortBy];
      return sortOrder === 'desc' ? valueB - valueA : valueA - valueB;
    });

    return results;
  }, [data.results, searchTerm, selectedAgent, selectedMetric, sortBy, sortOrder, showOnlyOutliers]);

  const paginatedResults = useMemo(() => {
    const startIndex = (currentPage - 1) * itemsPerPage;
    return filteredResults.slice(startIndex, startIndex + itemsPerPage);
  }, [filteredResults, currentPage, itemsPerPage]);

  const totalPages = Math.ceil(filteredResults.length / itemsPerPage);

  // Enhanced AI Explanation and Suggestion Handler
  const handleAIAnalysisAndSuggestion = async (result: EvaluationResult) => {
    setCurrentSuggestionResult(result);
    setShowAISuggestionModal(true);
    setIsLoadingAISuggestion(true);
    setCurrentAISuggestion(null);

    try {
      // Call the AI analysis endpoint
      const response = await fetch('http://127.0.0.1:8000/api/explain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          result: result,
          metric: 'overall_score',
          context: {
            agentId: result.agent_id,
            promptId: result.prompt_id,
            overallScore: result.metrics.overall_score,
            timestamp: result.generated_at,
            metricScore: result.metrics.overall_score
          }
        })
      });

      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.statusText}`);
      }

      const analysisData = await response.json();
      
      // Call prompt improvement endpoint
      const promptResponse = await fetch('http://127.0.0.1:8000/api/prompt-improvement', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: result.prompt,
          response: result.response,
          reference: result.reference,
          scores: result.metrics
        })
      });

      let promptSuggestion = null;
      if (promptResponse.ok) {
        promptSuggestion = await promptResponse.json();
      }
      
      // Structure the response data
      const suggestion: AISuggestion = {
        explanation: analysisData.explanation || 'No explanation available',
        suggestedPrompt: promptSuggestion?.suggestions?.improved_prompt || result.prompt,
        analysis: promptSuggestion?.suggestions?.analysis || 'No analysis available',
        expectedImprovements: promptSuggestion?.suggestions?.expected_improvements ? 
          (typeof promptSuggestion.suggestions.expected_improvements === 'string' 
            ? [promptSuggestion.suggestions.expected_improvements]
            : Array.isArray(promptSuggestion.suggestions.expected_improvements) 
              ? promptSuggestion.suggestions.expected_improvements
              : ['General improvements expected']
          ) : ['No specific improvements identified'],
        targetMetrics: Object.entries(result.metrics)
          .filter(([_, score]) => score < 0.7)
          .map(([metric]) => metric)
      };

      setCurrentAISuggestion(suggestion);

    } catch (error) {
      console.error('AI analysis failed:', error);
      setCurrentAISuggestion({
        explanation: 'Failed to generate AI explanation. Please check your API configuration and try again.',
        suggestedPrompt: result.prompt,
        analysis: 'Analysis unavailable due to API error.',
        expectedImprovements: ['Unable to generate suggestions at this time'],
        targetMetrics: []
      });
    } finally {
      setIsLoadingAISuggestion(false);
    }
  };

  const handleGetExplanation = async (result: EvaluationResult, metric: string) => {
    const key = `${result.prompt_id}_${result.agent_id}_${metric}`;
    
    if (explanations[key] || !onGetExplanation || loadingExplanations.has(key)) return;

    setLoadingExplanations(prev => new Set([...prev, key]));
    
    try {
      const explanation = await onGetExplanation(result, metric);
      setExplanations(prev => ({ ...prev, [key]: explanation }));
    } catch (error) {
      console.error('Failed to get explanation:', error);
      setExplanations(prev => ({ ...prev, [key]: 'Failed to generate explanation. Please try again.' }));
    } finally {
      setLoadingExplanations(prev => {
        const newSet = new Set(prev);
        newSet.delete(key);
        return newSet;
      });
    }
  };

  const toggleRowExpansion = (rowId: string) => {
    setExpandedRows(prev => {
      const newSet = new Set(prev);
      if (newSet.has(rowId)) {
        newSet.delete(rowId);
      } else {
        newSet.add(rowId);
      }
      return newSet;
    });
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const agents = ['all', ...Object.keys(data.agent_summaries)];

  return (
    <>
      <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
        {/* Header */}
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between mb-6 space-y-4 lg:space-y-0">
          <div className="flex items-center space-x-3">
            <List className="w-6 h-6 text-indigo-600" />
            <div>
              <h3 className="text-xl font-semibold text-gray-900">Detailed Results Analysis</h3>
              <p className="text-sm text-gray-600">
                {filteredResults.length} of {data.total_evaluated} results
                {showOnlyOutliers && ' (outliers only)'}
              </p>
            </div>
          </div>
          
          {/* Controls */}
          <div className="flex flex-wrap items-center gap-3">
            {/* Search */}
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
              <input
                type="text"
                placeholder="Search prompts, responses, IDs..."
                value={searchTerm}
                onChange={(e) => {
                  setSearchTerm(e.target.value);
                  setCurrentPage(1);
                }}
                className="pl-10 pr-4 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 w-64"
              />
            </div>
            
            {/* Agent Filter */}
            <select
              value={selectedAgent}
              onChange={(e) => {
                setSelectedAgent(e.target.value);
                setCurrentPage(1);
              }}
              className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
            >
              {agents.map(agent => (
                <option key={agent} value={agent}>
                  {agent === 'all' ? 'All Agents' : agent}
                </option>
              ))}
            </select>
            
            {/* Metric Selection for Outliers */}
            <select
              value={selectedMetric}
              onChange={(e) => setSelectedMetric(e.target.value as keyof Metrics)}
              className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
            >
              {Object.entries(metricLabels).map(([key, label]) => (
                <option key={key} value={key}>{label}</option>
              ))}
            </select>
            
            {/* Sort By */}
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as keyof Metrics)}
              className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
            >
              {Object.entries(metricLabels).map(([key, label]) => (
                <option key={key} value={key}>Sort by {label}</option>
              ))}
            </select>
            
            {/* Sort Order */}
            <button
              onClick={() => setSortOrder(sortOrder === 'desc' ? 'asc' : 'desc')}
              className="px-3 py-2 border border-gray-300 rounded-lg text-sm hover:bg-gray-50 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 flex items-center space-x-1"
            >
              {sortOrder === 'desc' ? <SortDesc className="w-4 h-4" /> : <SortAsc className="w-4 h-4" />}
              <span>{sortOrder === 'desc' ? 'High to Low' : 'Low to High'}</span>
            </button>
            
            {/* Outlier Toggle */}
            <button
              onClick={() => {
                setShowOnlyOutliers(!showOnlyOutliers);
                setCurrentPage(1);
              }}
              className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center space-x-2 ${
                showOnlyOutliers 
                  ? 'bg-red-100 text-red-800 border border-red-200' 
                  : 'bg-gray-100 text-gray-600 border border-gray-200'
              }`}
            >
              <AlertTriangle className="w-4 h-4" />
              <span>Outliers Only</span>
            </button>
          </div>
        </div>

        {/* Results Table */}
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Prompt Details
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Agent
                </th>
                {Object.entries(metricLabels).map(([key, label]) => (
                  <th key={key} className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider min-w-[100px]">
                    <div className="flex flex-col items-center space-y-1">
                      {React.createElement(metricIcons[key as keyof Metrics], { className: "w-4 h-4" })}
                      <span className="leading-tight">{label.split(' ').join(' ')}</span>
                    </div>
                  </th>
                ))}
                <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {paginatedResults.map((result, index) => {
                const rowId = `${result.prompt_id}_${result.agent_id}`;
                const isExpanded = expandedRows.has(rowId);
                const isEvenRow = index % 2 === 0;
                
                return (
                  <React.Fragment key={rowId}>
                    <tr className={`${isEvenRow ? 'bg-white' : 'bg-gray-50'} hover:bg-indigo-50 transition-colors`}>
                      {/* Prompt Details */}
                      <td className="px-4 py-4">
                        <div className="max-w-xs">
                          <div className="font-medium text-gray-900 truncate" title={result.prompt}>
                            {result.prompt.length > 60 ? `${result.prompt.substring(0, 60)}...` : result.prompt}
                          </div>
                          <div className="text-xs text-gray-500 mt-1">
                            ID: {result.prompt_id}
                          </div>
                          <div className="text-xs text-gray-400">
                            {new Date(result.generated_at).toLocaleDateString()}
                          </div>
                        </div>
                      </td>
                      
                      {/* Agent */}
                      <td className="px-4 py-4">
                        <div className="flex items-center">
                          <div className="w-8 h-8 rounded-full bg-indigo-100 flex items-center justify-center mr-3">
                            <span className="text-sm font-semibold text-indigo-600">
                              {result.agent_id.charAt(0).toUpperCase()}
                            </span>
                          </div>
                          <span className="font-medium text-gray-900">{result.agent_id}</span>
                        </div>
                      </td>
                      
                      {/* Metrics */}
                      {Object.entries(metricLabels).map(([key]) => {
                        const score = result.metrics[key as keyof Metrics];
                        const IconComponent = metricIcons[key as keyof Metrics];
                        const explanationKey = `${result.prompt_id}_${result.agent_id}_${key}`;
                        const hasExplanation = explanations[explanationKey];
                        const isLoadingExplanation = loadingExplanations.has(explanationKey);
                        
                        return (
                          <td key={key} className="px-4 py-4 text-center">
                            <div className={`inline-flex items-center space-x-2 px-3 py-2 rounded-full text-sm font-medium border ${getScoreColor(score)}`}>
                              <IconComponent className="w-4 h-4" />
                              <span>{(score * 100).toFixed(0)}%</span>
                            </div>
                            
                            {onGetExplanation && (
                              <div className="mt-2">
                                <button
                                  onClick={() => handleGetExplanation(result, key)}
                                  disabled={isLoadingExplanation}
                                  className={`text-xs px-2 py-1 rounded-full transition-colors ${
                                    hasExplanation 
                                      ? 'bg-green-100 text-green-700 hover:bg-green-200' 
                                      : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                                  }`}
                                  title={hasExplanation ? 'View explanation' : 'Get AI explanation'}
                                >
                                  {isLoadingExplanation ? (
                                    <RefreshCw className="w-3 h-3 animate-spin" />
                                  ) : (
                                    <MessageCircle className="w-3 h-3" />
                                  )}
                                </button>
                              </div>
                            )}
                          </td>
                        );
                      })}
                      
                      {/* Enhanced Actions */}
                      <td className="px-4 py-4 text-center">
                        <div className="flex items-center justify-center space-x-2">
                          {/* AI Analysis & Suggestion Button - NEW */}
                          <button
                            onClick={() => handleAIAnalysisAndSuggestion(result)}
                            className="group flex items-center space-x-1 px-3 py-2 bg-gradient-to-r from-purple-500 to-indigo-600 text-white rounded-lg hover:from-purple-600 hover:to-indigo-700 transition-all duration-200 transform hover:scale-105 shadow-md hover:shadow-lg"
                            title="Get AI Analysis & Prompt Suggestions"
                          >
                            <Bot className="w-4 h-4" />
                            <Sparkles className="w-3 h-3 opacity-75 group-hover:opacity-100" />
                            <span className="text-xs font-medium">AI Analyze</span>
                          </button>

                          <button
                            onClick={() => toggleRowExpansion(rowId)}
                            className="text-indigo-600 hover:text-indigo-800 transition-colors p-1 rounded-full hover:bg-indigo-100"
                            title={isExpanded ? 'Collapse details' : 'Expand details'}
                          >
                            {isExpanded ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
                          </button>
                          
                          <button
                            onClick={() => copyToClipboard(JSON.stringify(result, null, 2))}
                            className="text-gray-600 hover:text-gray-800 transition-colors p-1 rounded-full hover:bg-gray-100"
                            title="Copy result data"
                          >
                            <Copy className="w-4 h-4" />
                          </button>
                        </div>
                      </td>
                    </tr>
                    
                    {/* Expanded Row Details */}
                    {isExpanded && (
                      <tr>
                        <td colSpan={Object.keys(metricLabels).length + 3} className="px-4 py-6 bg-gray-50">
                          <div className="space-y-6">
                            {/* Text Content */}
                            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                              <div className="bg-white p-4 rounded-lg border border-gray-200">
                                <div className="flex items-center justify-between mb-2">
                                  <h4 className="font-semibold text-gray-900 flex items-center">
                                    <FileText className="w-4 h-4 mr-2" />
                                    Prompt
                                  </h4>
                                  <button
                                    onClick={() => copyToClipboard(result.prompt)}
                                    className="text-gray-400 hover:text-gray-600"
                                  >
                                    <Copy className="w-4 h-4" />
                                  </button>
                                </div>
                                <div className="text-sm text-gray-700 max-h-40 overflow-y-auto">
                                  {result.prompt}
                                </div>
                              </div>
                              
                              <div className="bg-white p-4 rounded-lg border border-gray-200">
                                <div className="flex items-center justify-between mb-2">
                                  <h4 className="font-semibold text-gray-900 flex items-center">
                                    <Brain className="w-4 h-4 mr-2" />
                                    Agent Response
                                  </h4>
                                  <button
                                    onClick={() => copyToClipboard(result.response)}
                                    className="text-gray-400 hover:text-gray-600"
                                  >
                                    <Copy className="w-4 h-4" />
                                  </button>
                                </div>
                                <div className="text-sm text-gray-700 max-h-40 overflow-y-auto">
                                  {result.response}
                                </div>
                              </div>
                              
                              <div className="bg-white p-4 rounded-lg border border-gray-200">
                                <div className="flex items-center justify-between mb-2">
                                  <h4 className="font-semibold text-gray-900 flex items-center">
                                    <Target className="w-4 h-4 mr-2" />
                                    Reference Answer
                                  </h4>
                                  <button
                                    onClick={() => copyToClipboard(result.reference)}
                                    className="text-gray-400 hover:text-gray-600"
                                  >
                                    <Copy className="w-4 h-4" />
                                  </button>
                                </div>
                                <div className="text-sm text-gray-700 max-h-40 overflow-y-auto">
                                  {result.reference}
                                </div>
                              </div>
                            </div>
                            
                            {/* AI Explanations */}
                            <div className="bg-white p-4 rounded-lg border border-gray-200">
                              <h4 className="font-semibold text-gray-900 mb-4 flex items-center">
                                <Brain className="w-4 h-4 mr-2" />
                                AI Explanations
                              </h4>
                              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                                {Object.entries(metricLabels).map(([key, label]) => {
                                  const explanationKey = `${result.prompt_id}_${result.agent_id}_${key}`;
                                  const explanation = explanations[explanationKey];
                                  const score = result.metrics[key as keyof Metrics];
                                  const IconComponent = metricIcons[key as keyof Metrics];
                                  
                                  return (
                                    <div key={key} className="bg-gray-50 p-3 rounded-lg">
                                      <div className="flex items-center justify-between mb-2">
                                        <div className="flex items-center space-x-2">
                                          <IconComponent className="w-4 h-4 text-indigo-600" />
                                          <span className="text-sm font-medium text-gray-700">{label}</span>
                                        </div>
                                        <span className={`text-xs px-2 py-1 rounded-full font-medium ${getScoreColor(score)}`}>
                                          {(score * 100).toFixed(0)}%
                                        </span>
                                      </div>
                                      <div className="text-xs text-gray-600">
                                        {explanation ? (
                                          <div className="space-y-1">
                                            <p>{explanation}</p>
                                            <button
                                              onClick={() => copyToClipboard(explanation)}
                                              className="text-indigo-600 hover:text-indigo-800 flex items-center space-x-1"
                                            >
                                              <Copy className="w-3 h-3" />
                                              <span>Copy</span>
                                            </button>
                                          </div>
                                        ) : (
                                          <div className="italic text-gray-400">
                                            Click the explanation button above to get AI insights
                                          </div>
                                        )}
                                      </div>
                                    </div>
                                  );
                                })}
                              </div>
                            </div>
                          </div>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between mt-6">
            <div className="text-sm text-gray-700">
              Showing {((currentPage - 1) * itemsPerPage) + 1} to {Math.min(currentPage * itemsPerPage, filteredResults.length)} of {filteredResults.length} results
            </div>
            
            <div className="flex items-center space-x-2">
              <button
                onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                disabled={currentPage === 1}
                className="px-3 py-1 border border-gray-300 rounded-md text-sm hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Previous
              </button>
              
              <div className="flex items-center space-x-1">
                {Array.from({ length: Math.min(7, totalPages) }, (_, i) => {
                  let pageNum;
                  if (totalPages <= 7) {
                    pageNum = i + 1;
                  } else if (currentPage <= 4) {
                    pageNum = i + 1;
                  } else if (currentPage >= totalPages - 3) {
                    pageNum = totalPages - 6 + i;
                  } else {
                    pageNum = currentPage - 3 + i;
                  }
                  
                  return (
                    <button
                      key={pageNum}
                      onClick={() => setCurrentPage(pageNum)}
                      className={`px-3 py-1 text-sm rounded-md ${
                        currentPage === pageNum 
                          ? 'bg-indigo-600 text-white' 
                          : 'border border-gray-300 hover:bg-gray-50'
                      }`}
                    >
                      {pageNum}
                    </button>
                  );
                })}
              </div>
              
              <button
                onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
                disabled={currentPage === totalPages}
                className="px-3 py-1 border border-gray-300 rounded-md text-sm hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Next
              </button>
            </div>
          </div>
        )}
      </div>

      {/* AI Suggestion Modal */}
      <AISuggestionModal
        isOpen={showAISuggestionModal}
        onClose={() => setShowAISuggestionModal(false)}
        result={currentSuggestionResult}
        suggestion={currentAISuggestion}
        isLoading={isLoadingAISuggestion}
      />
    </>
  );
};

const ScoreDistribution: React.FC<{ data: DashboardData }> = ({ data }) => {
  const [selectedMetric, setSelectedMetric] = useState<keyof Metrics>('overall_score');
  const [viewType, setViewType] = useState<'histogram' | 'boxplot' | 'violin'>('histogram');
  const [groupBy, setGroupBy] = useState<'all' | 'agent'>('all');

  const distributionData = useMemo(() => {
    if (groupBy === 'agent') {
      // Group by agent
      const agentData: any = {};
      data.results.forEach(result => {
        if (!agentData[result.agent_id]) {
          agentData[result.agent_id] = [];
        }
        agentData[result.agent_id].push(result.metrics[selectedMetric]);
      });
      
      return Object.entries(agentData).map(([agent, scores]: [string, any]) => {
        const sortedScores = scores.sort((a: number, b: number) => a - b);
        const q1 = sortedScores[Math.floor(sortedScores.length * 0.25)];
        const median = sortedScores[Math.floor(sortedScores.length * 0.5)];
        const q3 = sortedScores[Math.floor(sortedScores.length * 0.75)];
        const mean = scores.reduce((a: number, b: number) => a + b, 0) / scores.length;
        
        return {
          agent,
          scores,
          q1,
          median,
          q3,
          mean,
          min: Math.min(...scores),
          max: Math.max(...scores),
          count: scores.length
        };
      });
    } else {
      // All scores together
      const scores = data.results.map(result => result.metrics[selectedMetric]);
      
      if (viewType === 'histogram') {
        const bins = 20;
        const min = Math.min(...scores);
        const max = Math.max(...scores);
        const binWidth = (max - min) / bins;
        
        return Array.from({ length: bins }, (_, i) => {
          const binStart = min + i * binWidth;
          const binEnd = binStart + binWidth;
          const count = scores.filter(score => score >= binStart && (i === bins - 1 ? score <= binEnd : score < binEnd)).length;
          return {
            range: `${(binStart * 100).toFixed(0)}-${(binEnd * 100).toFixed(0)}%`,
            count,
            binStart,
            binEnd,
            density: count / scores.length / binWidth
          };
        });
      }
      
      return scores.map((score, index) => ({ index, score, agent: data.results[index].agent_id }));
    }
  }, [data.results, selectedMetric, viewType, groupBy]);

  const statistics = useMemo(() => {
    const scores = data.results.map(result => result.metrics[selectedMetric]);
    const sortedScores = [...scores].sort((a, b) => a - b);
    const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
    const variance = scores.reduce((acc, score) => acc + Math.pow(score - mean, 2), 0) / scores.length;
    const stdDev = Math.sqrt(variance);
    const outliers = findOutliers(scores);
    
    return {
      mean,
      median: sortedScores[Math.floor(sortedScores.length / 2)],
      stdDev,
      min: Math.min(...scores),
      max: Math.max(...scores),
      outlierCount: outliers.length,
      skewness: scores.reduce((acc, score) => acc + Math.pow((score - mean) / stdDev, 3), 0) / scores.length
    };
  }, [data.results, selectedMetric]);

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <BarChart3 className="w-6 h-6 text-indigo-600" />
          <div>
            <h3 className="text-xl font-semibold text-gray-900">Score Distribution Analysis</h3>
            <p className="text-sm text-gray-600">Statistical analysis of performance metrics</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-4">
          <select
            value={groupBy}
            onChange={(e) => setGroupBy(e.target.value as 'all' | 'agent')}
            className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
          >
            <option value="all">All Together</option>
            <option value="agent">By Agent</option>
          </select>
          
          <select
            value={viewType}
            onChange={(e) => setViewType(e.target.value as 'histogram' | 'boxplot' | 'violin')}
            className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
          >
            <option value="histogram">Histogram</option>
            <option value="boxplot">Box Plot</option>
            <option value="violin">Violin Plot</option>
          </select>
          
          <select
            value={selectedMetric}
            onChange={(e) => setSelectedMetric(e.target.value as keyof Metrics)}
            className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
          >
            {Object.entries(metricLabels).map(([key, label]) => (
              <option key={key} value={key}>{label}</option>
            ))}
          </select>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Chart */}
        <div className="lg:col-span-3">
          <ResponsiveContainer width="100%" height={400}>
            {viewType === 'histogram' && groupBy === 'all' ? (
              <BarChart data={distributionData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range" angle={-45} textAnchor="end" height={80} />
                <YAxis />
                <Tooltip 
                  formatter={(value: number) => [value, 'Count']}
                  labelFormatter={(label) => `Score Range: ${label}`}
                />
                <Bar dataKey="count" fill="#8884d8" radius={[2, 2, 0, 0]} />
              </BarChart>
            ) : viewType === 'boxplot' && groupBy === 'agent' ? (
              <ComposedChart data={distributionData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="agent" />
                <YAxis tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} />
                <Tooltip 
                  formatter={(value: number) => [`${(value * 100).toFixed(1)}%`, 'Score']}
                />
                <Bar dataKey="q1" fill="#f3f4f6" />
                <Bar dataKey="median" fill="#8884d8" />
                <Bar dataKey="q3" fill="#f3f4f6" />
                <Line type="monotone" dataKey="mean" stroke="#ff7c7c" strokeWidth={2} dot={{ r: 4 }} />
              </ComposedChart>
            ) : (
              <ScatterChart data={distributionData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey={groupBy === 'agent' ? 'agent' : 'index'} 
                  type={groupBy === 'agent' ? 'category' : 'number'}
                />
                <YAxis 
                  dataKey="score" 
                  tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                />
                <Tooltip 
                  formatter={(value: number) => [`${(value * 100).toFixed(1)}%`, 'Score']}
                />
                <Scatter dataKey="score" fill="#8884d8" />
              </ScatterChart>
            )}
          </ResponsiveContainer>
        </div>
        
        {/* Statistics Panel */}
        <div className="space-y-4">
          <h4 className="font-semibold text-gray-900">Statistics</h4>
          
          <div className="space-y-3">
            <div className="bg-gray-50 rounded-lg p-3">
              <div className="text-sm font-medium text-gray-700">Mean</div>
              <div className="text-lg font-semibold text-gray-900">
                {(statistics.mean * 100).toFixed(1)}%
              </div>
            </div>
            
            <div className="bg-gray-50 rounded-lg p-3">
              <div className="text-sm font-medium text-gray-700">Median</div>
              <div className="text-lg font-semibold text-gray-900">
                {(statistics.median * 100).toFixed(1)}%
              </div>
            </div>
            
            <div className="bg-gray-50 rounded-lg p-3">
              <div className="text-sm font-medium text-gray-700">Std Deviation</div>
              <div className="text-lg font-semibold text-gray-900">
                {(statistics.stdDev * 100).toFixed(1)}%
              </div>
            </div>
            
            <div className="bg-gray-50 rounded-lg p-3">
              <div className="text-sm font-medium text-gray-700">Range</div>
              <div className="text-lg font-semibold text-gray-900">
                {(statistics.min * 100).toFixed(1)}% - {(statistics.max * 100).toFixed(1)}%
              </div>
            </div>
            
            <div className="bg-gray-50 rounded-lg p-3">
              <div className="text-sm font-medium text-gray-700">Outliers</div>
              <div className="text-lg font-semibold text-gray-900 flex items-center space-x-2">
                <span>{statistics.outlierCount}</span>
                {statistics.outlierCount > 0 && (
                  <AlertTriangle className="w-4 h-4 text-orange-500" />
                )}
              </div>
            </div>
            
            <div className="bg-gray-50 rounded-lg p-3">
              <div className="text-sm font-medium text-gray-700">Skewness</div>
              <div className="text-lg font-semibold text-gray-900">
                {statistics.skewness.toFixed(2)}
                {Math.abs(statistics.skewness) > 0.5 && (
                  <span className="text-xs text-orange-600 ml-1">
                    ({statistics.skewness > 0 ? 'Right' : 'Left'} skewed)
                  </span>
                )}
              </div>
            </div>
          </div>
          
          <div className="pt-4 border-t border-gray-200">
            <div className="text-sm text-gray-600">
              <p className="mb-2">
                <strong>Interpretation:</strong>
              </p>
              {statistics.stdDev < 0.1 ? (
                <p className="text-green-600">Low variance - consistent performance</p>
              ) : statistics.stdDev > 0.2 ? (
                <p className="text-red-600">High variance - inconsistent performance</p>
              ) : (
                <p className="text-yellow-600">Moderate variance - some variation</p>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Enhanced Leaderboard Component
const EnhancedLeaderboard: React.FC<{ data: DashboardData }> = ({ data }) => {
  const [selectedMetric, setSelectedMetric] = useState<keyof Metrics>('overall_score');
  const [viewMode, setViewMode] = useState<'ranking' | 'comparison'>('ranking');

  const leaderboardData = useMemo(() => {
    const agentPerformance = Object.entries(data.agent_summaries).map(([agentId, metrics]) => {
      // Calculate additional insights
      const agentResults = data.results.filter(r => r.agent_id === agentId);
      const scores = agentResults.map(r => r.metrics[selectedMetric]);
      const variance = calculateVariance(scores);
      const outliers = findOutliers(scores);
      
      return {
        agentId,
        score: metrics[selectedMetric],
        metrics,
        consistency: 1 - variance, // Higher consistency = lower variance
        outlierCount: outliers.length,
        totalResponses: agentResults.length,
        bestScore: Math.max(...scores),
        worstScore: Math.min(...scores),
        improvement: scores.length > 1 ? scores[scores.length - 1] - scores[0] : 0
      };
    });

    return agentPerformance.sort((a, b) => b.score - a.score);
  }, [data.agent_summaries, data.results, selectedMetric]);

  const getBadge = (rank: number) => {
    switch (rank) {
      case 0: return { 
        icon: Trophy, 
        color: 'text-yellow-600 bg-yellow-50 border-yellow-200', 
        label: '🥇 Champion',
        description: 'Top performer'
      };
      case 1: return { 
        icon: Award, 
        color: 'text-gray-600 bg-gray-50 border-gray-200', 
        label: '🥈 Runner-up',
        description: 'Second place'
      };
      case 2: return { 
        icon: Award, 
        color: 'text-orange-600 bg-orange-50 border-orange-200', 
        label: '🥉 Third place',
        description: 'Bronze medalist'
      };
      default: return { 
        icon: Target, 
        color: 'text-indigo-600 bg-indigo-50 border-indigo-200', 
        label: `#${rank + 1}`,
        description: `${rank + 1}th place`
      };
    }
  };

  const getConsistencyBadge = (consistency: number) => {
    if (consistency > 0.9) return { label: 'Very Consistent', color: 'bg-green-100 text-green-800' };
    if (consistency > 0.8) return { label: 'Consistent', color: 'bg-blue-100 text-blue-800' };
    if (consistency > 0.7) return { label: 'Moderate', color: 'bg-yellow-100 text-yellow-800' };
    return { label: 'Variable', color: 'bg-red-100 text-red-800' };
  };

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <Trophy className="w-6 h-6 text-indigo-600" />
          <div>
            <h3 className="text-xl font-semibold text-gray-900">Performance Leaderboard</h3>
            <p className="text-sm text-gray-600">Ranked performance across all agents</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-4">
          <select
            value={viewMode}
            onChange={(e) => setViewMode(e.target.value as 'ranking' | 'comparison')}
            className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
          >
            <option value="ranking">Ranking View</option>
            <option value="comparison">Comparison View</option>
          </select>
          
          <select
            value={selectedMetric}
            onChange={(e) => setSelectedMetric(e.target.value as keyof Metrics)}
            className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
          >
            {Object.entries(metricLabels).map(([key, label]) => (
              <option key={key} value={key}>{label}</option>
            ))}
          </select>
        </div>
      </div>

      {viewMode === 'ranking' ? (
        <div className="space-y-4">
          {leaderboardData.map((agent, index) => {
            const badge = getBadge(index);
            const consistencyBadge = getConsistencyBadge(agent.consistency);
            const IconComponent = badge.icon;
            const grade = getPerformanceGrade(agent.score);
            
            return (
              <motion.div
                key={agent.agentId}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                className="flex items-center justify-between p-6 rounded-xl border border-gray-200 hover:shadow-lg transition-all duration-300 bg-gradient-to-r from-white to-gray-50"
              >
                <div className="flex items-center space-x-6">
                  {/* Rank Badge */}
                  <div className={`p-4 rounded-xl border ${badge.color} flex-shrink-0`}>
                    <IconComponent className="w-6 h-6" />
                  </div>
                  
                  {/* Agent Info */}
                  <div>
                    <div className="flex items-center space-x-3 mb-2">
                      <h4 className="text-xl font-bold text-gray-900">{agent.agentId}</h4>
                      <span className={`px-3 py-1 rounded-full text-sm font-medium ${grade.bg} ${grade.color}`}>
                        Grade {grade.grade}
                      </span>
                    </div>
                    <div className="flex items-center space-x-4 text-sm text-gray-600">
                      <span>{badge.label}</span>
                      <span>•</span>
                      <span>{agent.totalResponses} responses</span>
                      <span>•</span>
                      <span className={`px-2 py-1 rounded-full text-xs ${consistencyBadge.color}`}>
                        {consistencyBadge.label}
                      </span>
                    </div>
                  </div>
                </div>
                
                {/* Performance Metrics */}
                <div className="flex items-center space-x-8">
                  {/* Main Score */}
                  <div className="text-center">
                    <div className="text-3xl font-bold text-gray-900">
                      {(agent.score * 100).toFixed(1)}%
                    </div>
                    <div className="text-sm text-gray-600">
                      {metricLabels[selectedMetric]}
                    </div>
                  </div>
                  
                  {/* Additional Metrics */}
                  <div className="grid grid-cols-3 gap-4">
                    <div className="text-center">
                      <div className="text-lg font-semibold text-green-600">
                        {(agent.bestScore * 100).toFixed(1)}%
                      </div>
                      <div className="text-xs text-gray-500">Best</div>
                    </div>
                    <div className="text-center">
                      <div className="text-lg font-semibold text-red-600">
                        {(agent.worstScore * 100).toFixed(1)}%
                      </div>
                      <div className="text-xs text-gray-500">Worst</div>
                    </div>
                    <div className="text-center">
                      <div className={`text-lg font-semibold ${agent.improvement >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {agent.improvement >= 0 ? '+' : ''}{(agent.improvement * 100).toFixed(1)}%
                      </div>
                      <div className="text-xs text-gray-500">Trend</div>
                    </div>
                  </div>
                  
                  {/* Mini Metrics */}
                  <div className="flex space-x-2">
                    {Object.entries(metricLabels).filter(([key]) => key !== selectedMetric).slice(0, 3).map(([key]) => {
                      const score = agent.metrics[key as keyof Metrics];
                      const IconComp = metricIcons[key as keyof Metrics];
                      return (
                        <div key={key} className={`p-2 rounded-lg ${getScoreColor(score)} border`} title={metricLabels[key as keyof Metrics]}>
                          <IconComp className="w-4 h-4" />
                        </div>
                      );
                    })}
                  </div>
                </div>
              </motion.div>
            );
          })}
        </div>
      ) : (
        // Comparison View
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-3 text-left font-semibold text-gray-900">Agent</th>
                <th className="px-4 py-3 text-center font-semibold text-gray-900">Score</th>
                <th className="px-4 py-3 text-center font-semibold text-gray-900">Consistency</th>
                <th className="px-4 py-3 text-center font-semibold text-gray-900">Range</th>
                <th className="px-4 py-3 text-center font-semibold text-gray-900">Outliers</th>
                <th className="px-4 py-3 text-center font-semibold text-gray-900">Grade</th>
              </tr>
            </thead>
            <tbody>
              {leaderboardData.map((agent, index) => {
                const grade = getPerformanceGrade(agent.score);
                const consistencyBadge = getConsistencyBadge(agent.consistency);
                
                return (
                  <tr key={agent.agentId} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                    <td className="px-4 py-4">
                      <div className="flex items-center space-x-3">
                        <div className="w-8 h-8 rounded-full bg-indigo-100 flex items-center justify-center font-semibold text-indigo-600 text-sm">
                          {index + 1}
                        </div>
                        <span className="font-medium text-gray-900">{agent.agentId}</span>
                      </div>
                    </td>
                    <td className="px-4 py-4 text-center">
                      <span className="text-lg font-semibold text-gray-900">
                        {(agent.score * 100).toFixed(1)}%
                      </span>
                    </td>
                    <td className="px-4 py-4 text-center">
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${consistencyBadge.color}`}>
                        {(agent.consistency * 100).toFixed(1)}%
                      </span>
                    </td>
                    <td className="px-4 py-4 text-center">
                      <div className="text-sm">
                        <div className="text-green-600">{(agent.bestScore * 100).toFixed(1)}%</div>
                        <div className="text-red-600">{(agent.worstScore * 100).toFixed(1)}%</div>
                      </div>
                    </td>
                    <td className="px-4 py-4 text-center">
                      {agent.outlierCount > 0 ? (
                        <div className="flex items-center justify-center space-x-1">
                          <AlertTriangle className="w-4 h-4 text-orange-500" />
                          <span className="text-orange-600">{agent.outlierCount}</span>
                        </div>
                      ) : (
                        <span className="text-green-600">0</span>
                      )}
                    </td>
                    <td className="px-4 py-4 text-center">
                      <span className={`px-3 py-1 rounded-full text-sm font-medium ${grade.bg} ${grade.color}`}>
                        {grade.grade}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};


// Enhanced Export Panel Component (simplified for this version)
const EnhancedExportPanel: React.FC<{ 
  data: DashboardData;
  onExport?: (type: 'pdf' | 'excel' | 'csv') => Promise<void>;
}> = ({ data, onExport }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [isExporting, setIsExporting] = useState<string | null>(null);

  const exportOptions = [
    { 
      type: 'pdf' as const, 
      label: 'PDF Report', 
      icon: FileText, 
      description: 'Comprehensive report with charts and analysis',
      size: '~2-5 MB',
      features: ['Executive summary', 'All visualizations', 'Detailed tables', 'AI insights']
    },
    { 
      type: 'excel' as const, 
      label: 'Excel Workbook', 
      icon: Grid3x3, 
      description: 'Multiple sheets with raw data and summaries',
      size: '~500 KB - 2 MB',
      features: ['Raw data', 'Agent summaries', 'Pivot-ready format', 'Statistics']
    },
    { 
      type: 'csv' as const, 
      label: 'CSV Data', 
      icon: DownloadIcon, 
      description: 'Raw evaluation data for custom analysis',
      size: '~50-500 KB',
      features: ['All metrics', 'Timestamps', 'Prompt/response text', 'Machine readable']
    }
  ];

  const handleExport = async (type: 'pdf' | 'excel' | 'csv') => {
    if (!onExport) return;
    
    setIsExporting(type);
    try {
      await onExport(type);
    } catch (error) {
      console.error('Export failed:', error);
    } finally {
      setIsExporting(null);
      setIsOpen(false);
    }
  };

  const stats = {
    totalResults: data.total_evaluated,
    totalAgents: data.overall_summary.total_agents,
    avgScore: data.overall_summary.avg_overall_score,
    dataSize: `~${Math.ceil(data.total_evaluated * 0.5)}KB`
  };

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center space-x-3 px-6 py-3 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-xl hover:from-indigo-700 hover:to-purple-700 transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-105"
      >
        <Download className="w-5 h-5" />
        <span className="font-medium">Export Report</span>
        <ChevronDown className={`w-4 h-4 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: -10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -10, scale: 0.95 }}
            className="absolute right-0 top-full mt-3 w-96 bg-white rounded-xl shadow-2xl border border-gray-200 z-50 overflow-hidden"
          >
            <div className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h4 className="text-lg font-semibold text-gray-900">Export Options</h4>
                <button
                  onClick={() => setIsOpen(false)}
                  className="text-gray-400 hover:text-gray-600 transition-colors"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
              
              {/* Quick Stats */}
              <div className="bg-gray-50 rounded-lg p-4 mb-6">
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div className="text-center">
                    <div className="font-semibold text-gray-900">{stats.totalResults}</div>
                    <div className="text-gray-600">Results</div>
                  </div>
                  <div className="text-center">
                    <div className="font-semibold text-gray-900">{stats.totalAgents}</div>
                    <div className="text-gray-600">Agents</div>
                  </div>
                  <div className="text-center">
                    <div className="font-semibold text-gray-900">{(stats.avgScore * 100).toFixed(1)}%</div>
                    <div className="text-gray-600">Avg Score</div>
                  </div>
                  <div className="text-center">
                    <div className="font-semibold text-gray-900">{stats.dataSize}</div>
                    <div className="text-gray-600">Data Size</div>
                  </div>
                </div>
              </div>

              <div className="space-y-3">
                {exportOptions.map((option) => {
                  const isCurrentlyExporting = isExporting === option.type;
                  
                  return (
                    <motion.button
                      key={option.type}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      onClick={() => handleExport(option.type)}
                      disabled={isCurrentlyExporting || Boolean(isExporting)}
                      className="w-full flex items-start space-x-4 p-4 rounded-lg border border-gray-200 hover:border-indigo-300 hover:bg-indigo-50 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed text-left"
                    >
                      <div className="p-2 rounded-lg bg-indigo-100 flex-shrink-0">
                        {isCurrentlyExporting ? (
                          <RefreshCw className="w-5 h-5 text-indigo-600 animate-spin" />
                        ) : (
                          <option.icon className="w-5 h-5 text-indigo-600" />
                        )}
                      </div>
                      
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between mb-1">
                          <h5 className="font-medium text-gray-900">{option.label}</h5>
                          <span className="text-xs text-gray-500">{option.size}</span>
                        </div>
                        <p className="text-sm text-gray-600 mb-2">{option.description}</p>
                        
                        <div className="flex flex-wrap gap-1">
                          {option.features.map((feature, index) => (
                            <span key={index} className="px-2 py-1 bg-gray-100 text-xs text-gray-600 rounded">
                              {feature}
                            </span>
                          ))}
                        </div>
                      </div>
                    </motion.button>
                  );
                })}
              </div>
              
              <div className="mt-6 pt-4 border-t border-gray-200">
                <div className="text-xs text-gray-500 space-y-1">
                  <p>• Reports include all evaluation data and visualizations</p>
                  <p>• Exported files contain timestamp and metadata</p>
                  <p>• Large datasets may take longer to generate</p>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      
      {/* Backdrop */}
      {isOpen && (
        <div 
          className="fixed inset-0 z-40"
          onClick={() => setIsOpen(false)}
        />
      )}
    </div>
  );
};

// Main Comprehensive Dashboard Component
const ComprehensiveDashboard: React.FC<ComprehensiveDashboardProps> = ({
  data,
  onExport,
  onGetExplanation,
  evaluationId,
  onSave
}) => {
  const [activeSection, setActiveSection] = useState('overview');
  const [isLoading, setIsLoading] = useState(false);

  const handleExport = useCallback(async (type: 'pdf' | 'excel' | 'csv') => {
    if (!onExport) return;
    
    setIsLoading(true);
    try {
      await onExport(type);
    } catch (error) {
      console.error('Export failed:', error);
    } finally {
      setIsLoading(false);
    }
  }, [onExport]);

  const sections = [
    { id: 'overview', label: 'Overview', icon: BarChart3 },
    { id: 'analysis', label: 'Analysis', icon: RadarIcon },
    { id: 'details', label: 'Details', icon: List },
    { id: 'leaderboard', label: 'Rankings', icon: Trophy }
  ];

  const insights = useMemo(() => {
    const agents = Object.entries(data.agent_summaries);
    const overallScores = data.results.map(r => r.metrics.overall_score);
    const variance = calculateVariance(overallScores);
    const outliers = findOutliers(overallScores);
    
    return {
      topPerformer: agents.reduce((best, [id, metrics]) => 
        metrics.overall_score > best.score ? { id, score: metrics.overall_score } : best,
        { id: '', score: 0 }
      ),
      consistencyLevel: variance < 0.01 ? 'High' : variance < 0.05 ? 'Medium' : 'Low',
      outlierCount: outliers.length,
      averageGrade: getPerformanceGrade(data.overall_summary.avg_overall_score).grade
    };
  }, [data]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-indigo-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="p-2 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-lg">
                <Activity className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">AI Agent Evaluation Dashboard</h1>
                <div className="flex items-center space-x-6 text-sm text-gray-600 mt-1">
                  <span>{data.total_evaluated} evaluations</span>
                  <span>•</span>
                  <span>{data.overall_summary.total_agents} agents</span>
                  <span>•</span>
                  <span>Grade {insights.averageGrade}</span>
                  <span>•</span>
                  <span className={`px-2 py-1 rounded-full text-xs ${
                    insights.consistencyLevel === 'High' ? 'bg-green-100 text-green-800' :
                    insights.consistencyLevel === 'Medium' ? 'bg-yellow-100 text-yellow-800' :
                    'bg-red-100 text-red-800'
                  }`}>
                    {insights.consistencyLevel} Consistency
                  </span>
                  {evaluationId && (
                    <>
                      <span>•</span>
                      <span className="text-green-600 font-medium flex items-center space-x-1">
                        <Bookmark className="w-3 h-3" />
                        <span>Saved</span>
                      </span>
                    </>
                  )}
                </div>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              {/* Save Button (only for new evaluations) */}
              {!evaluationId && onSave && (
                <button
                  onClick={onSave}
                  className="flex items-center space-x-2 px-4 py-2 bg-green-100 text-green-700 rounded-lg hover:bg-green-200 transition-colors border border-green-200"
                >
                  <Bookmark className="w-4 h-4" />
                  <span>Save</span>
                </button>
              )}
              
              <EnhancedExportPanel data={data} onExport={handleExport} />
            </div>
          </div>
          
          {/* Navigation */}
          <div className="flex items-center space-x-1 mt-6 border-b border-gray-200">
            {sections.map((section) => (
              <button
                key={section.id}
                onClick={() => setActiveSection(section.id)}
                className={`flex items-center space-x-2 px-4 py-3 text-sm font-medium rounded-t-lg transition-colors ${
                  activeSection === section.id
                    ? 'bg-indigo-50 text-indigo-600 border-b-2 border-indigo-600'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                }`}
              >
                <section.icon className="w-4 h-4" />
                <span>{section.label}</span>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-7xl mx-auto px-6 py-8">
        {isLoading && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white rounded-lg p-6 flex items-center space-x-3">
              <RefreshCw className="w-5 h-5 animate-spin text-indigo-600" />
              <span className="text-gray-900">Generating export...</span>
            </div>
          </div>
        )}

        {activeSection === 'overview' && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-8"
          >
            <SummaryCards data={data} />
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
              <AgentRadarChart data={data} />
              <MetricBarChart data={data} />
            </div>
          </motion.div>
        )}

        {activeSection === 'analysis' && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-8"
          >
            <MetricHeatmap data={data} />
            <ScoreDistribution data={data} />
          </motion.div>
        )}

        {activeSection === 'details' && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <DetailedResultsTable data={data} onGetExplanation={onGetExplanation} />
          </motion.div>
        )}

        {activeSection === 'leaderboard' && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <EnhancedLeaderboard data={data} />
          </motion.div>
        )}
      </div>
    </div>
  );
};

export default ComprehensiveDashboard;

// Export types for external use
export type { 
  Metrics, 
  EvaluationResult, 
  DashboardData, 
  ComprehensiveDashboardProps 
};

// Export utility functions
export { 
  getScoreColor, 
  getPerformanceGrade, 
  calculateVariance, 
  findOutliers 
};