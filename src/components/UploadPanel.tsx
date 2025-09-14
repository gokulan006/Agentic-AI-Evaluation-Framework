import React, { useState, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Upload, FileText, X, RefreshCw, CheckCircle, AlertTriangle, 
  Settings, Bot, Wand2, MessageCircle, Lightbulb, 
  ToggleLeft, ToggleRight, Eye, EyeOff, Brain, Sparkles,
  Download, HelpCircle, FileJson, Code, Layers
} from 'lucide-react';

interface AIFeatures {
  autoReference: boolean;
  detailedExplanations: boolean;
  promptSuggestions: boolean;
  assistantChat: boolean;
}

interface UploadPanelProps {
  isOpen: boolean;
  onClose: () => void;
  onUploaded: (result: any) => void;
  onError: (error: string) => void;
  evaluationMode: 'traditional' | 'ai_assisted' | 'hybrid';
  onEvaluationModeChange: (mode: 'traditional' | 'ai_assisted' | 'hybrid') => void;
  aiFeatures: AIFeatures;
  onAIFeaturesChange: (features: AIFeatures) => void;
}

const UploadPanel: React.FC<UploadPanelProps> = ({
  isOpen,
  onClose,
  onUploaded,
  onError,
  evaluationMode,
  onEvaluationModeChange,
  aiFeatures,
  onAIFeaturesChange
}) => {
  const [isUploading, setIsUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [file, setFile] = useState<File | null>(null);
  const [previewData, setPreviewData] = useState<any>(null);
  const [showPreview, setShowPreview] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFiles = useCallback(async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    
    const selectedFile = files[0];
    
    if (!selectedFile.name.endsWith('.json')) {
      onError('Please upload a JSON file containing evaluation data');
      return;
    }
    
    setFile(selectedFile);
    
    // Preview file content
    try {
      const text = await selectedFile.text();
      const data = JSON.parse(text);
      setPreviewData(data);
      setShowPreview(true);
    } catch (error) {
      onError('Invalid JSON file format');
    }
  }, [onError]);

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    handleFiles(e.dataTransfer.files);
  }, [handleFiles]);

  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
  }, []);

  const handleUpload = useCallback(async () => {
    if (!file) {
      onError('Please select a file first');
      return;
    }

    setIsUploading(true);
    setUploadProgress(0);

    try {
      const formData = new FormData();
      formData.append('file', file);

      // Simulate progress for better UX
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => Math.min(prev + 10, 90));
      }, 200);

      const response = await fetch(`http://127.0.0.1:8000/api/upload-and-evaluate?mode=${evaluationMode}`, {
        method: 'POST',
        body: formData,
      });

      clearInterval(progressInterval);
      setUploadProgress(100);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(`Upload failed: ${response.statusText} - ${errorData.detail || 'Unknown error'}`);
      }

      const result = await response.json();

      if (!result.success) {
        throw new Error(result.message || 'Upload failed');
      }

      onUploaded(result);
      
      // Reset state
      setFile(null);
      setPreviewData(null);
      setShowPreview(false);
      setUploadProgress(0);
      
    } catch (error) {
      console.error('Upload error:', error);
      onError(error instanceof Error ? error.message : 'Upload failed');
      setUploadProgress(0);
    } finally {
      setIsUploading(false);
    }
  }, [file, evaluationMode, onUploaded, onError]);

  const toggleAIFeature = (feature: keyof AIFeatures) => {
    onAIFeaturesChange({
      ...aiFeatures,
      [feature]: !aiFeatures[feature]
    });
  };

  const downloadSampleFile = () => {
    const sampleData = [
      {
        "prompt_id": "sample_1",
        "prompt": "Explain the concept of machine learning in simple terms",
        "agent_id": "gpt-4",
        "response": "Machine learning is a type of artificial intelligence where computers learn to make predictions or decisions by finding patterns in data, rather than being explicitly programmed with rules.",
        "reference": "Machine learning is a subset of AI that enables systems to automatically learn and improve from experience without being explicitly programmed..."
      },
      {
        "prompt_id": "sample_2", 
        "prompt": "What are the benefits of renewable energy?",
        "agent_id": "claude-3",
        "response": "Renewable energy sources like solar and wind offer environmental benefits by reducing greenhouse gas emissions, economic advantages through job creation, and energy security by reducing dependence on fossil fuel imports.",
        "reference": "Renewable energy provides multiple benefits including environmental protection, economic growth, energy independence, and sustainability..."
      }
    ];

    const blob = new Blob([JSON.stringify(sampleData, null, 2)], { type: 'application/json' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'sample-evaluation-data.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  };

  const modeDescriptions = {
    traditional: 'Uses your trained BERT model for fast evaluation',
    ai_assisted: 'AI-powered evaluation with detailed analysis (requires OpenAI API)',
    hybrid: 'Combines traditional model with AI insights for balanced results'
  };

  const modeIcons = {
    traditional: Brain,
    ai_assisted: Bot,
    hybrid: Layers
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
          <div className="flex items-center justify-between p-6 border-b border-gray-200 bg-gradient-to-r from-indigo-50 to-purple-50">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-lg">
                <Upload className="w-6 h-6 text-white" />
              </div>
              <div>
                <h2 className="text-2xl font-bold text-gray-900">Upload Evaluation Data</h2>
                <p className="text-sm text-gray-600">
                  Upload JSON data for AI-enhanced evaluation with detailed analysis
                </p>
              </div>
            </div>
            
            <div className="flex items-center space-x-2">
              <button
                onClick={() => setShowSettings(!showSettings)}
                className={`p-2 rounded-lg transition-colors ${
                  showSettings ? 'bg-indigo-100 text-indigo-600' : 'text-gray-400 hover:text-gray-600'
                }`}
                title="Evaluation Settings"
              >
                <Settings className="w-5 h-5" />
              </button>
              <button
                onClick={onClose}
                className="p-2 text-gray-400 hover:text-gray-600 transition-colors rounded-lg hover:bg-gray-100"
              >
                <X className="w-6 h-6" />
              </button>
            </div>
          </div>

          <div className="flex max-h-[calc(90vh-80px)]">
            {/* Settings Sidebar */}
            <AnimatePresence>
              {showSettings && (
                <motion.div
                  initial={{ width: 0, opacity: 0 }}
                  animate={{ width: 'auto', opacity: 1 }}
                  exit={{ width: 0, opacity: 0 }}
                  className="border-r border-gray-200 bg-gray-50 p-6 overflow-y-auto"
                  style={{ minWidth: '320px' }}
                >
                  <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                    <Bot className="w-5 h-5 mr-2 text-purple-600" />
                    AI Settings
                  </h3>

                  {/* Evaluation Mode */}
                  <div className="mb-6">
                    <label className="block text-sm font-medium text-gray-700 mb-3">
                      Evaluation Mode
                    </label>
                    <div className="space-y-3">
                      {(['traditional', 'ai_assisted', 'hybrid'] as const).map(mode => {
                        const Icon = modeIcons[mode];
                        return (
                          <label key={mode} className="flex items-start space-x-3 cursor-pointer p-3 rounded-lg hover:bg-white transition-colors">
                            <input
                              type="radio"
                              value={mode}
                              checked={evaluationMode === mode}
                              onChange={(e) => onEvaluationModeChange(e.target.value as any)}
                              className="mt-1 text-indigo-600 border-gray-300 focus:ring-indigo-500"
                            />
                            <div className="flex-1">
                              <div className="flex items-center space-x-2 mb-1">
                                <Icon className="w-4 h-4 text-indigo-600" />
                                <span className="text-sm font-medium text-gray-900 capitalize">
                                  {mode.replace('_', ' ')}
                                </span>
                                {mode === 'ai_assisted' && (
                                  <span className="px-2 py-1 bg-purple-100 text-purple-700 text-xs rounded-full">
                                    AI-Powered
                                  </span>
                                )}
                              </div>
                              <div className="text-xs text-gray-500">
                                {modeDescriptions[mode]}
                              </div>
                            </div>
                          </label>
                        );
                      })}
                    </div>
                  </div>

                  {/* AI Features */}
                  {(evaluationMode === 'ai_assisted' || evaluationMode === 'hybrid') && (
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-3">
                        AI Enhancement Features
                      </label>
                      <div className="space-y-3">
                        <div className="flex items-center justify-between p-3 bg-white rounded-lg">
                          <div className="flex items-center space-x-2">
                            <Wand2 className="w-4 h-4 text-purple-600" />
                            <div>
                              <div className="text-sm font-medium text-gray-700">Auto Reference</div>
                              <div className="text-xs text-gray-500">Generate reference answers with AI</div>
                            </div>
                          </div>
                          <button
                            onClick={() => toggleAIFeature('autoReference')}
                            className={`p-1 rounded-full transition-colors ${
                              aiFeatures.autoReference ? 'text-green-600' : 'text-gray-400'
                            }`}
                          >
                            {aiFeatures.autoReference ? (
                              <ToggleRight className="w-5 h-5" />
                            ) : (
                              <ToggleLeft className="w-5 h-5" />
                            )}
                          </button>
                        </div>

                        <div className="flex items-center justify-between p-3 bg-white rounded-lg">
                          <div className="flex items-center space-x-2">
                            <MessageCircle className="w-4 h-4 text-blue-600" />
                            <div>
                              <div className="text-sm font-medium text-gray-700">Detailed Explanations</div>
                              <div className="text-xs text-gray-500">AI explanations for each score</div>
                            </div>
                          </div>
                          <button
                            onClick={() => toggleAIFeature('detailedExplanations')}
                            className={`p-1 rounded-full transition-colors ${
                              aiFeatures.detailedExplanations ? 'text-green-600' : 'text-gray-400'
                            }`}
                          >
                            {aiFeatures.detailedExplanations ? (
                              <ToggleRight className="w-5 h-5" />
                            ) : (
                              <ToggleLeft className="w-5 h-5" />
                            )}
                          </button>
                        </div>

                        <div className="flex items-center justify-between p-3 bg-white rounded-lg">
                          <div className="flex items-center space-x-2">
                            <Lightbulb className="w-4 h-4 text-yellow-600" />
                            <div>
                              <div className="text-sm font-medium text-gray-700">Prompt Suggestions</div>
                              <div className="text-xs text-gray-500">Get improvement recommendations</div>
                            </div>
                          </div>
                          <button
                            onClick={() => toggleAIFeature('promptSuggestions')}
                            className={`p-1 rounded-full transition-colors ${
                              aiFeatures.promptSuggestions ? 'text-green-600' : 'text-gray-400'
                            }`}
                          >
                            {aiFeatures.promptSuggestions ? (
                              <ToggleRight className="w-5 h-5" />
                            ) : (
                              <ToggleLeft className="w-5 h-5" />
                            )}
                          </button>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Info Panel */}
                  <div className="mt-6 p-4 bg-blue-50 rounded-lg">
                    <div className="flex items-start space-x-2">
                      <Sparkles className="w-4 h-4 text-blue-600 mt-0.5 flex-shrink-0" />
                      <div className="text-sm text-blue-800">
                        <p className="font-medium mb-1">AI-Enhanced Evaluation</p>
                        <ul className="text-xs space-y-1">
                          <li>• Comprehensive 7-metric analysis</li>
                          <li>• Automatic reference generation</li>
                          <li>• Detailed explanations for each score</li>
                          <li>• Prompt optimization suggestions</li>
                        </ul>
                      </div>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Main Content */}
            <div className="flex-1 p-6 overflow-y-auto">
              {!showPreview ? (
                <>
                  {/* Upload Area */}
                  <div
                    onDrop={handleDrop}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    className={`border-2 border-dashed rounded-2xl p-8 text-center transition-all duration-300 ${
                      dragActive 
                        ? 'border-indigo-500 bg-indigo-50' 
                        : 'border-gray-300 hover:border-indigo-400 hover:bg-indigo-50/50'
                    }`}
                  >
                    <div className="max-w-md mx-auto">
                      <div className={`w-20 h-20 mx-auto mb-6 rounded-full flex items-center justify-center transition-colors ${
                        dragActive ? 'bg-indigo-100' : 'bg-gray-100'
                      }`}>
                        <FileText className={`w-10 h-10 ${dragActive ? 'text-indigo-600' : 'text-gray-400'}`} />
                      </div>
                      
                      <h3 className="text-xl font-semibold text-gray-900 mb-2">
                        Upload Your Evaluation Data
                      </h3>
                      
                      <p className="text-gray-600 mb-6">
                        Drag and drop your JSON file here, or click to browse files
                      </p>
                      
                      <div className="space-y-4">
                        <input
                          ref={fileInputRef}
                          type="file"
                          accept=".json"
                          onChange={(e) => handleFiles(e.target.files)}
                          className="hidden"
                        />
                        
                        <button
                          onClick={() => fileInputRef.current?.click()}
                          className="inline-flex items-center px-6 py-3 bg-indigo-600 text-white font-medium rounded-xl hover:bg-indigo-700 transition-colors shadow-lg hover:shadow-xl"
                        >
                          <Upload className="w-5 h-5 mr-2" />
                          Choose File
                        </button>
                        
                        <div className="text-sm text-gray-500">
                          Supported format: JSON • Max size: 10MB
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Help Section */}
                  <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="p-4 bg-gray-50 rounded-lg">
                      <h4 className="font-semibold text-gray-900 mb-2 flex items-center">
                        <FileJson className="w-4 h-4 mr-2 text-indigo-600" />
                        Required JSON Format
                      </h4>
                      <div className="text-sm text-gray-600 space-y-1">
                        <div>• <span className="font-medium">prompt_id</span>: Unique identifier</div>
                        <div>• <span className="font-medium">prompt</span>: Question/instruction</div>
                        <div>• <span className="font-medium">agent_id</span>: AI model identifier</div>
                        <div>• <span className="font-medium">response</span>: Agent's response</div>
                        <div>• <span className="font-medium">reference</span>: Expected answer (optional with AI)</div>
                      </div>
                    </div>

                    <div className="p-4 bg-gray-50 rounded-lg">
                      <h4 className="font-semibold text-gray-900 mb-2 flex items-center">
                        <Download className="w-4 h-4 mr-2 text-green-600" />
                        Need Help?
                      </h4>
                      <div className="text-sm text-gray-600 mb-3">
                        Download a sample file to see the correct format
                      </div>
                      <button
                        onClick={downloadSampleFile}
                        className="inline-flex items-center px-3 py-2 bg-green-100 text-green-700 rounded-lg hover:bg-green-200 transition-colors text-sm"
                      >
                        <Download className="w-4 h-4 mr-2" />
                        Download Sample
                      </button>
                    </div>
                  </div>
                </>
              ) : (
                /* File Preview */
                <div>
                  <div className="flex items-center justify-between mb-6">
                    <div className="flex items-center space-x-3">
                      <CheckCircle className="w-6 h-6 text-green-600" />
                      <div>
                        <h3 className="text-lg font-semibold text-gray-900">File Preview</h3>
                        <p className="text-sm text-gray-600">
                          {file?.name} • {Array.isArray(previewData) ? previewData.length : 0} items
                        </p>
                      </div>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      <button
                        onClick={() => setShowPreview(false)}
                        className="px-3 py-2 text-gray-600 hover:text-gray-800 transition-colors"
                      >
                        Choose Different File
                      </button>
                    </div>
                  </div>

                  {/* Preview Content */}
                  <div className="bg-gray-50 rounded-lg p-4 mb-6 max-h-64 overflow-y-auto">
                    <pre className="text-sm text-gray-800 whitespace-pre-wrap">
                      {JSON.stringify(previewData, null, 2)}
                    </pre>
                  </div>

                  {/* Upload Progress */}
                  {isUploading && (
                    <div className="mb-6">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium text-gray-700">
                          Processing with {evaluationMode.replace('_', ' ')} mode...
                        </span>
                        <span className="text-sm text-gray-500">{uploadProgress}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-gradient-to-r from-indigo-500 to-purple-600 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${uploadProgress}%` }}
                        />
                      </div>
                    </div>
                  )}

                  {/* Upload Button */}
                  <div className="flex justify-end space-x-3">
                    <button
                      onClick={() => {
                        setFile(null);
                        setPreviewData(null);
                        setShowPreview(false);
                      }}
                      disabled={isUploading}
                      className="px-6 py-3 text-gray-700 border border-gray-300 rounded-xl hover:bg-gray-50 transition-colors disabled:opacity-50"
                    >
                      Cancel
                    </button>
                    
                    <button
                      onClick={handleUpload}
                      disabled={isUploading}
                      className="flex items-center space-x-2 px-8 py-3 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-xl hover:from-indigo-700 hover:to-purple-700 transition-all duration-300 shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {isUploading ? (
                        <>
                          <RefreshCw className="w-5 h-5 animate-spin" />
                          <span>Processing...</span>
                        </>
                      ) : (
                        <>
                          <Sparkles className="w-5 h-5" />
                          <span>Start Evaluation</span>
                        </>
                      )}
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

export default UploadPanel;