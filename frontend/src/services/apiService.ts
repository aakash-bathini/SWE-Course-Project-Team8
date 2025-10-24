import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor to include auth token
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Add response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Token expired or invalid
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export const apiService = {
  // Health check
  async getHealth() {
    const response = await apiClient.get('/health');
    return response.data;
  },

  // Model operations
  async uploadModel(file: File, metadata: any) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model_data', JSON.stringify(metadata));
    
    const response = await apiClient.post('/models/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  async rateModel(modelId: string) {
    const response = await apiClient.get(`/models/${modelId}/rate`);
    return response.data;
  },

  async downloadModel(modelId: string, aspect?: string) {
    const params = aspect ? { aspect } : {};
    const response = await apiClient.get(`/models/${modelId}/download`, { params });
    return response.data;
  },

  async deleteModel(modelId: string) {
    const response = await apiClient.delete(`/models/${modelId}`);
    return response.data;
  },

  async listModels(page = 1, pageSize = 10) {
    const response = await apiClient.get('/models', {
      params: { page, page_size: pageSize },
    });
    return response.data;
  },

  async ingestHuggingFaceModel(modelName: string) {
    const response = await apiClient.post('/models/ingest', null, {
      params: { model_name: modelName },
    });
    return response.data;
  },

  // User operations
  async registerUser(userData: any) {
    const response = await apiClient.post('/register', userData);
    return response.data;
  },

  async authenticateUser(credentials: any) {
    const response = await apiClient.post('/authenticate', credentials);
    return response.data;
  },
};

export default apiService;
