import React, { useState, useEffect, useCallback } from 'react';
import { motion} from 'framer-motion';
import { 
  History, Search, Clock, Target,
  Eye, Trash2, BarChart3, RefreshCw,
  Grid3x3, List, SortAsc, SortDesc, Plus, ArrowLeft, Bookmark
} from 'lucide-react';

interface StoredEvaluation {
  id: string;
  name: string;
  description?: string;
  created_at: string;
  total_evaluated: number;
  total_agents: number;
  avg_overall_score: number;
}

interface EvaluationHistoryProps {
  onLoadEvaluation: (id: string) => void;
  onBackToLanding: () => void;
  onNewEvaluation: () => void;
  evaluationStats?: any;
}

const EvaluationHistory: React.FC<EvaluationHistoryProps> = ({
  onLoadEvaluation,
  onBackToLanding,
  onNewEvaluation,
  evaluationStats
}) => {
  const [evaluations, setEvaluations] = useState<StoredEvaluation[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [sortBy, setSortBy] = useState<'created_at' | 'name' | 'avg_overall_score'>('created_at');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [selectedEvaluations, setSelectedEvaluations] = useState<Set<string>>(new Set());
  const [currentPage, setCurrentPage] = useState(0);
  const [totalCount, setTotalCount] = useState(0);
  const itemsPerPage = 12;

  // Fetch evaluations
  const fetchEvaluations = useCallback(async () => {
    setIsLoading(true);
    try {
      const response = await fetch(
        `http://127.0.0.1:8000/api/evaluations?limit=${itemsPerPage}&offset=${currentPage * itemsPerPage}`
      );
      
      if (response.ok) {
        const result = await response.json();
        setEvaluations(result.evaluations);
        setTotalCount(result.pagination.total);
      } else {
        console.error('Failed to fetch evaluations');
      }
    } catch (error) {
      console.error('Error fetching evaluations:', error);
    } finally {
      setIsLoading(false);
    }
  }, [currentPage, itemsPerPage]);

  useEffect(() => {
    fetchEvaluations();
  }, [fetchEvaluations]);

  // Filter and sort evaluations
  const filteredEvaluations = evaluations
    .filter(evaluation => 
      evaluation.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      (evaluation.description && evaluation.description.toLowerCase().includes(searchTerm.toLowerCase()))
    )
    .sort((a, b) => {
      let valueA, valueB;
      
      switch (sortBy) {
        case 'name':
          valueA = a.name.toLowerCase();
          valueB = b.name.toLowerCase();
          break;
        case 'avg_overall_score':
          valueA = a.avg_overall_score;
          valueB = b.avg_overall_score;
          break;
        case 'created_at':
        default:
          valueA = new Date(a.created_at).getTime();
          valueB = new Date(b.created_at).getTime();
          break;
      }
      
      if (sortOrder === 'asc') {
        return valueA > valueB ? 1 : -1;
      } else {
        return valueA < valueB ? 1 : -1;
      }
    });

  // Delete evaluation
  const handleDeleteEvaluation = async (id: string, name: string) => {
    if (!confirm(`Are you sure you want to delete "${name}"? This action cannot be undone.`)) {
      return;
    }
    
    try {
      const response = await fetch(`http://127.0.0.1:8000/api/evaluations/${id}`, {
        method: 'DELETE'
      });
      
      if (response.ok) {
        fetchEvaluations();
        setSelectedEvaluations(prev => {
          const newSet = new Set(prev);
          newSet.delete(id);
          return newSet;
        });
      } else {
        alert('Failed to delete evaluation');
      }
    } catch (error) {
      console.error('Error deleting evaluation:', error);
      alert('Error deleting evaluation');
    }
  };

  // Bulk delete
  const handleBulkDelete = async () => {
    if (selectedEvaluations.size === 0) return;
    
    if (!confirm(`Are you sure you want to delete ${selectedEvaluations.size} evaluation(s)? This action cannot be undone.`)) {
      return;
    }
    
    const deletePromises = Array.from(selectedEvaluations).map(id =>
      fetch(`http://127.0.0.1:8000/api/evaluations/${id}`, { method: 'DELETE' })
    );
    
    try {
      await Promise.all(deletePromises);
      fetchEvaluations();
      setSelectedEvaluations(new Set());
    } catch (error) {
      console.error('Error with bulk delete:', error);
      alert('Some evaluations could not be deleted');
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getPerformanceColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600 bg-green-100';
    if (score >= 0.6) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  const getPerformanceGrade = (score: number) => {
    if (score >= 0.9) return 'A+';
    if (score >= 0.8) return 'A';
    if (score >= 0.7) return 'B+';
    if (score >= 0.6) return 'B';
    if (score >= 0.5) return 'C';
    return 'D';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-indigo-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <button
                onClick={onBackToLanding}
                className="flex items-center space-x-2 px-4 py-2 text-gray-600 hover:text-gray-900 transition-colors"
              >
                <ArrowLeft className="w-5 h-5" />
                <span>Back</span>
              </button>
              
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-gradient-to-r from-purple-500 to-indigo-600 rounded-lg">
                  <History className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold text-gray-900">Evaluation History</h1>
                  <p className="text-sm text-gray-600">
                    {totalCount} evaluations analyzed
                  </p>
                </div>
              </div>
            </div>
            
            <button
              onClick={onNewEvaluation}
              className="flex items-center space-x-2 px-6 py-3 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-xl hover:from-indigo-700 hover:to-purple-700 transition-all duration-300 shadow-lg hover:shadow-xl"
            >
              <Plus className="w-5 h-5" />
              <span>New Evaluation</span>
            </button>
          </div>
        </div>
      </div>

      {/* Stats Overview */}
      {evaluationStats && (
        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-white rounded-xl shadow-lg p-6"
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">Total Evaluations</p>
                  <p className="text-3xl font-bold text-gray-900">{evaluationStats.total_evaluations}</p>
                </div>
                <Bookmark className="w-8 h-8 text-indigo-600" />
              </div>
            </motion.div>
            
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="bg-white rounded-xl shadow-lg p-6"
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">Total Responses</p>
                  <p className="text-3xl font-bold text-gray-900">{evaluationStats.total_responses}</p>
                </div>
                <BarChart3 className="w-8 h-8 text-green-600" />
              </div>
            </motion.div>
            
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="bg-white rounded-xl shadow-lg p-6"
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">Average Score</p>
                  <p className="text-3xl font-bold text-gray-900">
                    {(evaluationStats.avg_overall_score * 100).toFixed(1)}%
                  </p>
                </div>
                <Target className="w-8 h-8 text-yellow-600" />
              </div>
            </motion.div>
            
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="bg-white rounded-xl shadow-lg p-6"
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">Recent (30 days)</p>
                  <p className="text-3xl font-bold text-gray-900">{evaluationStats.recent_evaluations}</p>
                </div>
                <Clock className="w-8 h-8 text-purple-600" />
              </div>
            </motion.div>
          </div>
        </div>
      )}

      {/* Controls */}
      <div className="max-w-7xl mx-auto px-6 pb-8">
        <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between space-y-4 lg:space-y-0">
            {/* Search */}
            <div className="relative flex-1 max-w-md">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
              <input
                type="text"
                placeholder="Search evaluations..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
              />
            </div>
            
            {/* Controls */}
            <div className="flex items-center space-x-4">
              {/* Sort */}
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value as any)}
                className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
              >
                <option value="created_at">Date Created</option>
                <option value="name">Name</option>
                <option value="avg_overall_score">Performance</option>
              </select>
              
              <button
                onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
                className="p-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
              >
                {sortOrder === 'asc' ? <SortAsc className="w-5 h-5" /> : <SortDesc className="w-5 h-5" />}
              </button>
              
              {/* View Mode */}
              <div className="flex border border-gray-300 rounded-lg overflow-hidden">
                <button
                  onClick={() => setViewMode('grid')}
                  className={`p-2 ${viewMode === 'grid' ? 'bg-indigo-100 text-indigo-600' : 'text-gray-600 hover:bg-gray-50'}`}
                >
                  <Grid3x3 className="w-5 h-5" />
                </button>
                <button
                  onClick={() => setViewMode('list')}
                  className={`p-2 ${viewMode === 'list' ? 'bg-indigo-100 text-indigo-600' : 'text-gray-600 hover:bg-gray-50'}`}
                >
                  <List className="w-5 h-5" />
                </button>
              </div>
              
              {/* Bulk Actions */}
              {selectedEvaluations.size > 0 && (
                <button
                  onClick={handleBulkDelete}
                  className="flex items-center space-x-2 px-4 py-2 bg-red-100 text-red-700 rounded-lg hover:bg-red-200 transition-colors"
                >
                  <Trash2 className="w-4 h-4" />
                  <span>Delete ({selectedEvaluations.size})</span>
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Content */}
        {isLoading ? (
          <div className="flex items-center justify-center py-20">
            <RefreshCw className="w-8 h-8 animate-spin text-indigo-600" />
            <span className="ml-3 text-gray-600">Loading evaluations...</span>
          </div>
        ) : filteredEvaluations.length === 0 ? (
          <div className="text-center py-20">
            <History className="w-16 h-16 text-gray-300 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-gray-600 mb-2">No evaluations found</h3>
            <p className="text-gray-500 mb-6">
              {searchTerm ? 'Try adjusting your search terms' : 'Create your first evaluation to get started'}
            </p>
            <button
              onClick={onNewEvaluation}
              className="inline-flex items-center px-6 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
            >
              <Plus className="w-5 h-5 mr-2" />
              Create Evaluation
            </button>
          </div>
        ) : (
          <>
            {/* Evaluations Grid/List */}
            <div className={viewMode === 'grid' 
              ? 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6'
              : 'space-y-4'
            }>
              {filteredEvaluations.map((evaluation, index) => (
                <motion.div
                  key={evaluation.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.05 }}
                  className={`bg-white rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 overflow-hidden ${
                    viewMode === 'list' ? 'flex items-center' : ''
                  }`}
                >
                  {viewMode === 'grid' ? (
                    // Grid View
                    <div className="p-6">
                      <div className="flex items-start justify-between mb-4">
                        <div className="flex-1">
                          <h3 className="text-lg font-semibold text-gray-900 mb-1 line-clamp-2">
                            {evaluation.name}
                          </h3>
                          {evaluation.description && (
                            <p className="text-sm text-gray-600 line-clamp-2 mb-2">
                              {evaluation.description}
                            </p>
                          )}
                        </div>
                        
                        <input
                          type="checkbox"
                          checked={selectedEvaluations.has(evaluation.id)}
                          onChange={(e) => {
                            const newSet = new Set(selectedEvaluations);
                            if (e.target.checked) {
                              newSet.add(evaluation.id);
                            } else {
                              newSet.delete(evaluation.id);
                            }
                            setSelectedEvaluations(newSet);
                          }}
                          className="rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
                        />
                      </div>
                      
                      <div className="space-y-3 mb-4">
                        <div className="flex items-center justify-between text-sm">
                          <span className="text-gray-600">Performance</span>
                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${getPerformanceColor(evaluation.avg_overall_score)}`}>
                            Grade {getPerformanceGrade(evaluation.avg_overall_score)} • {(evaluation.avg_overall_score * 100).toFixed(1)}%
                          </span>
                        </div>
                        
                        <div className="flex items-center justify-between text-sm text-gray-600">
                          <span>Responses</span>
                          <span className="font-medium">{evaluation.total_evaluated}</span>
                        </div>
                        
                        <div className="flex items-center justify-between text-sm text-gray-600">
                          <span>Agents</span>
                          <span className="font-medium">{evaluation.total_agents}</span>
                        </div>
                        
                        <div className="flex items-center justify-between text-sm text-gray-600">
                          <span>Created</span>
                          <span>{formatDate(evaluation.created_at)}</span>
                        </div>
                      </div>
                      
                      <div className="flex space-x-2">
                        <button
                          onClick={() => onLoadEvaluation(evaluation.id)}
                          className="flex-1 flex items-center justify-center space-x-2 px-4 py-2 bg-indigo-100 text-indigo-700 rounded-lg hover:bg-indigo-200 transition-colors"
                        >
                          <Eye className="w-4 h-4" />
                          <span>View</span>
                        </button>
                        
                        <button
                          onClick={() => handleDeleteEvaluation(evaluation.id, evaluation.name)}
                          className="p-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  ) : (
                    // List View
                    <div className="flex items-center justify-between p-6 w-full">
                      <div className="flex items-center space-x-4">
                        <input
                          type="checkbox"
                          checked={selectedEvaluations.has(evaluation.id)}
                          onChange={(e) => {
                            const newSet = new Set(selectedEvaluations);
                            if (e.target.checked) {
                              newSet.add(evaluation.id);
                            } else {
                              newSet.delete(evaluation.id);
                            }
                            setSelectedEvaluations(newSet);
                          }}
                          className="rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
                        />
                        
                        <div className="flex-1">
                          <h3 className="text-lg font-semibold text-gray-900">{evaluation.name}</h3>
                          {evaluation.description && (
                            <p className="text-sm text-gray-600 mt-1">{evaluation.description}</p>
                          )}
                        </div>
                      </div>
                      
                      <div className="flex items-center space-x-8">
                        <div className="text-center">
                          <p className="text-sm font-medium text-gray-900">{evaluation.total_evaluated}</p>
                          <p className="text-xs text-gray-600">Responses</p>
                        </div>
                        
                        <div className="text-center">
                          <p className="text-sm font-medium text-gray-900">{evaluation.total_agents}</p>
                          <p className="text-xs text-gray-600">Agents</p>
                        </div>
                        
                        <div className="text-center">
                          <span className={`px-3 py-1 rounded-full text-sm font-medium ${getPerformanceColor(evaluation.avg_overall_score)}`}>
                            {(evaluation.avg_overall_score * 100).toFixed(1)}%
                          </span>
                        </div>
                        
                        <div className="text-center">
                          <p className="text-sm text-gray-600">{formatDate(evaluation.created_at)}</p>
                        </div>
                        
                        <div className="flex space-x-2">
                          <button
                            onClick={() => onLoadEvaluation(evaluation.id)}
                            className="flex items-center space-x-1 px-4 py-2 bg-indigo-100 text-indigo-700 rounded-lg hover:bg-indigo-200 transition-colors"
                          >
                            <Eye className="w-4 h-4" />
                            <span>View</span>
                          </button>
                          
                          <button
                            onClick={() => handleDeleteEvaluation(evaluation.id, evaluation.name)}
                            className="p-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      </div>
                    </div>
                  )}
                </motion.div>
              ))}
            </div>

            {/* Pagination */}
            {totalCount > itemsPerPage && (
              <div className="flex items-center justify-between mt-8">
                <p className="text-sm text-gray-600">
                  Showing {currentPage * itemsPerPage + 1} to {Math.min((currentPage + 1) * itemsPerPage, totalCount)} of {totalCount} evaluations
                </p>
                
                <div className="flex space-x-2">
                  <button
                    onClick={() => setCurrentPage(Math.max(0, currentPage - 1))}
                    disabled={currentPage === 0}
                    className="px-4 py-2 border border-gray-300 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50 transition-colors"
                  >
                    Previous
                  </button>
                  
                  <button
                    onClick={() => setCurrentPage(currentPage + 1)}
                    disabled={(currentPage + 1) * itemsPerPage >= totalCount}
                    className="px-4 py-2 border border-gray-300 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50 transition-colors"
                  >
                    Next
                  </button>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default EvaluationHistory;