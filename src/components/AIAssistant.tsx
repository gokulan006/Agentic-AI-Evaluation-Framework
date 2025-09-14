import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  X, Send, Bot, User, Loader2, Copy, ThumbsUp, ThumbsDown, 
  Sparkles, ChevronDown, ChevronUp, MoreVertical, Download, Trash2, ArrowDown
} from 'lucide-react';

interface AIMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  context?: any;
}

interface AIAssistantProps {
  isOpen: boolean;
  onClose: () => void;
  messages: AIMessage[];
  onSendMessage: (message: string) => Promise<void>;
  isTyping: boolean;
  sessionId: string;
  onClearConversation?: () => void;
  onExportConversation?: (messages: AIMessage[]) => void;
}

const AIAssistant: React.FC<AIAssistantProps> = ({
  isOpen,
  onClose,
  messages,
  onSendMessage,
  isTyping,
  sessionId,
  onClearConversation,
  onExportConversation
}) => {
  const [inputMessage, setInputMessage] = useState('');
  const [isMinimized, setIsMinimized] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [showQuickActions, setShowQuickActions] = useState(false);
  const [showScrollToBottom, setShowScrollToBottom] = useState(false);
  const [isUserScrolling, setIsUserScrolling] = useState(false);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll to bottom when new messages arrive (only if user isn't manually scrolling)
  useEffect(() => {
    if (!isUserScrolling) {
      scrollToBottom();
    }
  }, [messages, isTyping, isUserScrolling]);

  // Focus input when opened or restored from minimized
  useEffect(() => {
    if (isOpen && !isMinimized) {
      setTimeout(() => {
        inputRef.current?.focus();
      }, 100);
    }
  }, [isOpen, isMinimized]);

  // Close on ESC key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  // Handle scroll behavior and show/hide scroll-to-bottom button
  useEffect(() => {
    const container = messagesContainerRef.current;
    if (!container) return;

    const handleScroll = () => {
      const { scrollTop, scrollHeight, clientHeight } = container;
      const isAtBottom = scrollHeight - scrollTop - clientHeight < 10;
      
      setShowScrollToBottom(!isAtBottom && messages.length > 3);
      setIsUserScrolling(!isAtBottom);
    };

    container.addEventListener('scroll', handleScroll);
    return () => container.removeEventListener('scroll', handleScroll);
  }, [messages]);

  const scrollToBottom = (smooth = true) => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ 
        behavior: smooth ? 'smooth' : 'auto',
        block: 'end'
      });
    }
  };

  const handleScrollToBottom = () => {
    setIsUserScrolling(false);
    scrollToBottom();
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!inputMessage.trim() || isSending) return;
    
    const messageText = inputMessage.trim();
    setInputMessage('');
    setIsSending(true);
    
    // Auto-scroll when user sends a message
    setIsUserScrolling(false);
    
    try {
      await onSendMessage(messageText);
    } catch (error) {
      console.error('Failed to send message:', error);
    } finally {
      setIsSending(false);
    }
  };

  const handleQuickPrompt = async (prompt: string) => {
    if (isSending) return;
    
    setIsUserScrolling(false);
    setIsSending(true);
    try {
      await onSendMessage(prompt);
    } catch (error) {
      console.error('Failed to send quick prompt:', error);
    } finally {
      setIsSending(false);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  const clearConversation = () => {
    if (confirm('Are you sure you want to clear the conversation? This action cannot be undone.')) {
      if (onClearConversation) {
        onClearConversation();
      } else {
        // Fallback: reload the page to clear conversation
        window.location.reload();
      }
      setShowQuickActions(false);
    }
  };

  const exportConversation = () => {
    if (onExportConversation) {
      onExportConversation(messages);
    } else {
      // Fallback: export as JSON file
      const conversationData = {
        sessionId,
        exportedAt: new Date().toISOString(),
        messages: messages.map(msg => ({
          ...msg,
          content: msg.content.trim()
        }))
      };
      
      const dataStr = JSON.stringify(conversationData, null, 2);
      const dataBlob = new Blob([dataStr], { type: 'application/json' });
      
      const link = document.createElement('a');
      link.href = URL.createObjectURL(dataBlob);
      link.download = `ai-assistant-conversation-${sessionId}.json`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
    setShowQuickActions(false);
  };

  const quickPrompts = [
    "Explain the evaluation metrics",
    "How do I improve accuracy scores?",
    "What does the assumption score mean?",
    "Show me how to export results",
    "Help me understand the dashboard",
    "How does AI evaluation work?",
    "What are the best practices for prompts?",
    "Compare different evaluation modes"
  ];

  if (!isOpen) return null;

  return (
    <>
      {/* Backdrop for dropdown menus */}
      {showQuickActions && (
        <div 
          className="fixed inset-0 z-40 bg-transparent"
          onClick={() => setShowQuickActions(false)}
        />
      )}
      
      <AnimatePresence>
        <motion.div
          initial={{ opacity: 0, scale: 0.8, x: 50, y: 50 }}
          animate={{ opacity: 1, scale: 1, x: 0, y: 0 }}
          exit={{ opacity: 0, scale: 0.8, x: 50, y: 50 }}
          transition={{ type: "spring", damping: 25, stiffness: 300 }}
          className="fixed bottom-4 right-4 z-50 bg-white rounded-2xl shadow-2xl border border-gray-200 overflow-hidden flex flex-col"
          style={{
            width: isMinimized ? '350px' : '420px',
            height: isMinimized ? '70px' : '650px',
            maxWidth: 'calc(100vw - 2rem)',
            maxHeight: 'calc(100vh - 2rem)'
          }}
        >
          {/* Header - Always Visible */}
          <div className="relative bg-gradient-to-r from-indigo-500 via-purple-500 to-indigo-600 text-white">
            {/* Header Content */}
            <div className="flex items-center justify-between p-4">
              <div className="flex items-center space-x-3 flex-1 min-w-0">
                <div className="w-10 h-10 bg-white/20 backdrop-blur-sm rounded-full flex items-center justify-center shadow-lg flex-shrink-0">
                  <Bot className="w-5 h-5 text-white" />
                </div>
                <div className="flex-1 min-w-0">
                  <h3 className="font-semibold text-sm text-white truncate">AI Assistant</h3>
                  <p className="text-xs text-indigo-100 truncate">
                    {isTyping ? (
                      <span className="flex items-center space-x-1">
                        <span>Typing</span>
                        <div className="flex space-x-1">
                          <div className="w-1 h-1 bg-white rounded-full animate-bounce"></div>
                          <div className="w-1 h-1 bg-white rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                          <div className="w-1 h-1 bg-white rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                        </div>
                      </span>
                    ) : (
                      `${messages.length} messages • Ready to help`
                    )}
                  </p>
                </div>
              </div>
              
              <div className="flex items-center space-x-1 ml-2">
                {/* Quick Actions Menu */}
                <div className="relative">
                  <button
                    onClick={() => setShowQuickActions(!showQuickActions)}
                    className="p-2 hover:bg-white/20 rounded-full transition-colors"
                    title="Quick actions"
                  >
                    <MoreVertical className="w-4 h-4" />
                  </button>
                  
                  <AnimatePresence>
                    {showQuickActions && (
                      <motion.div
                        initial={{ opacity: 0, scale: 0.9, y: 10 }}
                        animate={{ opacity: 1, scale: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.9, y: 10 }}
                        className="absolute right-0 top-full mt-2 w-48 bg-white rounded-lg shadow-xl border border-gray-200 py-2 z-50"
                      >
                        <button
                          onClick={clearConversation}
                          className="w-full px-4 py-2 text-left text-sm text-gray-700 hover:bg-gray-100 flex items-center space-x-2 transition-colors"
                        >
                          <Trash2 className="w-4 h-4" />
                          <span>Clear conversation</span>
                        </button>
                        <button
                          onClick={exportConversation}
                          className="w-full px-4 py-2 text-left text-sm text-gray-700 hover:bg-gray-100 flex items-center space-x-2 transition-colors"
                        >
                          <Download className="w-4 h-4" />
                          <span>Export chat</span>
                        </button>
                        <div className="border-t border-gray-200 my-1"></div>
                        <button
                          onClick={() => setShowQuickActions(false)}
                          className="w-full px-4 py-2 text-left text-sm text-gray-500 hover:bg-gray-100 flex items-center space-x-2 transition-colors"
                        >
                          <X className="w-4 h-4" />
                          <span>Close menu</span>
                        </button>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
                
                {/* Minimize Button */}
                <button
                  onClick={() => setIsMinimized(!isMinimized)}
                  className="p-2 hover:bg-white/20 rounded-full transition-colors"
                  title={isMinimized ? 'Expand' : 'Minimize'}
                >
                  {isMinimized ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                </button>
                
                {/* Close Button */}
                <button
                  onClick={onClose}
                  className="p-2 hover:bg-red-500/20 hover:text-red-100 rounded-full transition-colors"
                  title="Close assistant"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Progress bar when typing */}
            {isTyping && (
              <div className="absolute bottom-0 left-0 w-full h-0.5 bg-white/20">
                <div className="h-full bg-white/60 animate-pulse"></div>
              </div>
            )}
          </div>

          {/* Content - Only show when not minimized */}
          <AnimatePresence>
            {!isMinimized && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.3 }}
                className="flex flex-col flex-1 overflow-hidden"
              >
                {/* Messages Container with Professional Scrolling */}
                <div className="flex-1 relative bg-gray-50 overflow-hidden">
                  <div
                    ref={messagesContainerRef}
                    className="absolute inset-0 overflow-y-auto p-4 space-y-4 scroll-smooth"
                    style={{
                      scrollbarWidth: 'thin',
                      scrollbarColor: '#cbd5e1 #f1f5f9'
                    }}
                  >
                    {/* Custom Scrollbar Styles */}
                    <style jsx >{`
                      div::-webkit-scrollbar {
                        width: 6px;
                      }
                      div::-webkit-scrollbar-track {
                        background: #f1f5f9;
                        border-radius: 3px;
                      }
                      div::-webkit-scrollbar-thumb {
                        background: #cbd5e1;
                        border-radius: 3px;
                      }
                      div::-webkit-scrollbar-thumb:hover {
                        background: #94a3b8;
                      }
                    `}</style>

                    {/* Message List */}
                    {messages.map((message, index) => (
                      <motion.div
                        key={message.id}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: index * 0.05 }}
                        className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                      >
                        <div className={`flex items-start max-w-[85%] ${message.role === 'user' ? 'flex-row-reverse' : ''}`}>
                          {/* Avatar */}
                          <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 shadow-sm mt-1 ${
                            message.role === 'user' 
                              ? 'bg-indigo-500 text-white ml-3' 
                              : message.role === 'assistant'
                              ? 'bg-purple-500 text-white mr-3'
                              : 'bg-blue-500 text-white mr-3'
                          }`}>
                            {message.role === 'user' ? (
                              <User className="w-4 h-4" />
                            ) : message.role === 'assistant' ? (
                              <Bot className="w-4 h-4" />
                            ) : (
                              <Sparkles className="w-4 h-4" />
                            )}
                          </div>
                          
                          {/* Message Bubble */}
                          <div className={`rounded-2xl px-4 py-3 shadow-sm max-w-full ${message.role === 'user'
                              ? 'bg-indigo-600 text-white'
                              : message.role === 'assistant'
                              ? 'bg-white text-gray-800 border border-gray-200'
                              : 'bg-blue-50 text-blue-800 border border-blue-200'
                          }`}>
                            <div className="text-sm leading-relaxed whitespace-pre-wrap break-words">
                              {message.content}
                            </div>
                            
                            <div className="flex items-center justify-between mt-3">
                              <span className={`text-xs ${message.role === 'user' 
                                  ? 'text-indigo-200' 
                                  : 'text-gray-500'
                              }`}>
                                {formatTimestamp(message.timestamp)}
                              </span>
                              
                              {message.role === 'assistant' && (
                                <div className="flex items-center space-x-1">
                                  <button
                                    onClick={() => copyToClipboard(message.content)}
                                    className="p-1 hover:bg-gray-200 rounded text-gray-500 hover:text-gray-700 transition-colors"
                                    title="Copy message"
                                  >
                                    <Copy className="w-3 h-3" />
                                  </button>
                                  <button
                                    className="p-1 hover:bg-gray-200 rounded text-gray-500 hover:text-green-600 transition-colors"
                                    title="Helpful"
                                  >
                                    <ThumbsUp className="w-3 h-3" />
                                  </button>
                                  <button
                                    className="p-1 hover:bg-gray-200 rounded text-gray-500 hover:text-red-600 transition-colors"
                                    title="Not helpful"
                                  >
                                    <ThumbsDown className="w-3 h-3" />
                                  </button>
                                </div>
                              )}
                            </div>
                          </div>
                        </div>
                      </motion.div>
                    ))}
                    
                    {/* Typing Indicator */}
                    {isTyping && (
                      <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="flex justify-start"
                      >
                        <div className="flex items-start">
                          <div className="w-8 h-8 bg-purple-500 text-white rounded-full flex items-center justify-center flex-shrink-0 mr-3">
                            <Bot className="w-4 h-4" />
                          </div>
                          <div className="bg-white border border-gray-200 rounded-2xl px-4 py-3 shadow-sm">
                            <div className="flex items-center space-x-2">
                              <div className="flex space-x-1">
                                <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce"></div>
                                <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                                <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                              </div>
                              <span className="text-sm text-gray-500">AI is thinking...</span>
                            </div>
                          </div>
                        </div>
                      </motion.div>
                    )}
                    
                    {/* Scroll anchor */}
                    <div ref={messagesEndRef} />
                  </div>
                  
                  {/* Scroll to bottom button */}
                  <AnimatePresence>
                    {showScrollToBottom && (
                      <motion.button
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.8 }}
                        onClick={handleScrollToBottom}
                        className="absolute right-4 bottom-4 w-8 h-8 bg-white border border-gray-200 rounded-full shadow-md flex items-center justify-center hover:bg-gray-50 transition-colors z-10"
                        title="Scroll to bottom"
                      >
                        <ArrowDown className="w-4 h-4 text-gray-600" />
                      </motion.button>
                    )}
                  </AnimatePresence>
                </div>

                {/* Quick Prompts */}
                <div className="bg-white border-t border-gray-200 p-3">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="text-xs font-medium text-gray-500 uppercase tracking-wider">Quick Prompts</h4>
                    <button
                      onClick={() => setShowQuickActions(!showQuickActions)}
                      className="text-xs text-indigo-600 hover:text-indigo-800 transition-colors"
                    >
                      More options
                    </button>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-2">
                    {quickPrompts.slice(0, 4).map((prompt, index) => (
                      <button
                        key={index}
                        onClick={() => handleQuickPrompt(prompt)}
                        disabled={isSending}
                        className="text-xs text-left bg-gray-100 hover:bg-gray-200 text-gray-700 px-3 py-2 rounded-lg transition-colors truncate disabled:opacity-50 disabled:cursor-not-allowed"
                        title={prompt}
                      >
                        {prompt}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Input Area */}
                <form onSubmit={handleSendMessage} className="p-4 bg-white border-t border-gray-200">
                  <div className="flex items-center space-x-2">
                    <input
                      ref={inputRef}
                      type="text"
                      value={inputMessage}
                      onChange={(e) => setInputMessage(e.target.value)}
                      placeholder="Type your message..."
                      className="flex-1 bg-gray-100 border border-gray-200 rounded-full px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all"
                      disabled={isSending}
                    />
                    
                    <button
                      type="submit"
                      disabled={!inputMessage.trim() || isSending}
                      className="bg-indigo-600 text-white p-3 rounded-full hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors shadow-sm flex-shrink-0"
                      title="Send message"
                    >
                      {isSending ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : (
                        <Send className="w-4 h-4" />
                      )}
                    </button>
                  </div>
                </form>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </AnimatePresence>
    </>
  );
};

export default AIAssistant;