import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Bookmark, FileText} from 'lucide-react';

interface SaveEvaluationModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (name: string, description?: string) => Promise<void>;
  evaluationData: any;
}

const SaveEvaluationModal: React.FC<SaveEvaluationModalProps> = ({
  isOpen,
  onClose,
  onSave,
  evaluationData
}) => {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!name.trim()) {
      setError('Evaluation name is required');
      return;
    }
    
    setIsSubmitting(true);
    setError('');
    
    try {
      await onSave(name.trim(), description.trim() || undefined);
      setName('');
      setDescription('');
      onClose();
    } catch (error) {
      setError('Failed to save evaluation');
    } finally {
      setIsSubmitting(false);
    }
  };

  const generateSuggestedName = () => {
    const date = new Date().toLocaleDateString();
    const agentCount = evaluationData?.overall_summary?.total_agents || 0;
    const totalEvals = evaluationData?.total_evaluated || 0;
    return `Evaluation ${date} - ${agentCount} agents, ${totalEvals} responses`;
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4"
        onClick={(e) => e.target === e.currentTarget && onClose()}
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.95 }}
          className="bg-white rounded-2xl shadow-2xl w-full max-w-md p-6"
        >
          {/* Header */}
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-green-100 rounded-lg">
                <Bookmark className="w-6 h-6 text-green-600" />
              </div>
              <div>
                <h3 className="text-xl font-semibold text-gray-900">Save Evaluation</h3>
                <p className="text-sm text-gray-600">Store this evaluation for future reference</p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600 transition-colors"
            >
              <X className="w-6 h-6" />
            </button>
          </div>

          {/* Evaluation Preview */}
          <div className="bg-gray-50 rounded-lg p-4 mb-6">
            <h4 className="font-medium text-gray-900 mb-3">Evaluation Summary</h4>
            <div className="grid grid-cols-3 gap-4 text-sm">
              <div className="text-center">
                <p className="font-medium text-gray-900">{evaluationData?.total_evaluated || 0}</p>
                <p className="text-gray-600">Responses</p>
              </div>
              <div className="text-center">
                <p className="font-medium text-gray-900">{evaluationData?.overall_summary?.total_agents || 0}</p>
                <p className="text-gray-600">Agents</p>
              </div>
              <div className="text-center">
                <p className="font-medium text-gray-900">
                  {evaluationData?.overall_summary?.avg_overall_score 
                    ? (evaluationData.overall_summary.avg_overall_score * 100).toFixed(1) + '%'
                    : 'N/A'
                  }
                </p>
                <p className="text-gray-600">Avg Score</p>
              </div>
            </div>
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-4">
            {/* Name Field */}
            <div>
              <label htmlFor="name" className="block text-sm font-medium text-gray-700 mb-2">
                Evaluation Name *
              </label>
              <input
                type="text"
                id="name"
                value={name}
                onChange={(e) => {
                  setName(e.target.value);
                  setError('');
                }}
                placeholder="Enter a descriptive name..."
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-green-500 transition-colors"
                maxLength={100}
              />
              <button
                type="button"
                onClick={() => setName(generateSuggestedName())}
                className="mt-2 text-sm text-green-600 hover:text-green-700 transition-colors"
              >
                Use suggested name
              </button>
            </div>

            {/* Description Field */}
            <div>
              <label htmlFor="description" className="block text-sm font-medium text-gray-700 mb-2">
                Description (Optional)
              </label>
              <textarea
                id="description"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Add notes about this evaluation..."
                rows={3}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-green-500 transition-colors resize-none"
                maxLength={500}
              />
              <p className="mt-1 text-xs text-gray-500">
                {description.length}/500 characters
              </p>
            </div>

            {/* Error Message */}
            {error && (
              <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
                <p className="text-red-800 text-sm">{error}</p>
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex space-x-3 pt-2">
              <button
                type="button"
                onClick={onClose}
                disabled={isSubmitting}
                className="flex-1 px-4 py-3 text-gray-700 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={!name.trim() || isSubmitting}
                className="flex-1 px-4 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed font-medium"
              >
                {isSubmitting ? 'Saving...' : 'Save Evaluation'}
              </button>
            </div>
          </form>

          {/* Info */}
          <div className="mt-6 p-3 bg-blue-50 rounded-lg">
            <div className="flex items-start space-x-2">
              <FileText className="w-4 h-4 text-blue-600 mt-0.5 flex-shrink-0" />
              <div className="text-sm text-blue-800">
                <p className="font-medium">What gets saved:</p>
                <ul className="mt-1 list-disc list-inside space-y-1 text-xs">
                  <li>All evaluation results and metrics</li>
                  <li>Agent performance summaries</li>
                  <li>Individual prompt-response pairs</li>
                  <li>Dashboard data for future viewing</li>
                </ul>
              </div>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

export default SaveEvaluationModal;
