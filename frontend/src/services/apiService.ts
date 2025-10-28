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
      // Ensure headers object exists and is mutable
      const headers: Record<string, string> = (config.headers as Record<string, string>) || {};
      // Spec uses X-Authorization; keep Authorization for compatibility
      headers['X-Authorization'] = token;
      // If token already includes 'bearer', pass through; else prefix
      const hasBearer = token.toLowerCase().startsWith('bearer ');
      headers.Authorization = hasBearer ? token : `Bearer ${token}`;
      config.headers = headers;
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

// Types matching OpenAPI spec
export interface ArtifactMetadata {
  name: string;
  id: string;
  type: 'model' | 'dataset' | 'code';
}

export interface ArtifactData {
  url: string;
}

export interface Artifact {
  metadata: ArtifactMetadata;
  data: ArtifactData;
}

export interface ArtifactQuery {
  name: string;
  types?: ('model' | 'dataset' | 'code')[];
}

export interface ModelRating {
  name: string;
  category: string;
  net_score: number;
  net_score_latency: number;
  ramp_up_time: number;
  ramp_up_time_latency: number;
  bus_factor: number;
  bus_factor_latency: number;
  performance_claims: number;
  performance_claims_latency: number;
  license: number;
  license_latency: number;
  dataset_and_code_score: number;
  dataset_and_code_score_latency: number;
  dataset_quality: number;
  dataset_quality_latency: number;
  code_quality: number;
  code_quality_latency: number;
  reproducibility: number;
  reproducibility_latency: number;
  reviewedness: number;
  reviewedness_latency: number;
  tree_score: number;
  tree_score_latency: number;
  size_score: {
    raspberry_pi: number;
    jetson_nano: number;
    desktop_pc: number;
    aws_server: number;
  };
  size_score_latency: number;
}

export interface ArtifactCost {
  total_cost: number;
  standalone_cost?: number;
}

export interface AuthenticationRequest {
  user: {
    name: string;
    is_admin: boolean;
  };
  secret: {
    password: string;
  };
}

// AuthenticationToken is a string per spec; backend returns a raw string token
export type AuthenticationToken = string;

export const apiService = {
  // Health check
  async getHealth() {
    const response = await apiClient.get('/health');
    return response.data;
  },

  // Authentication
  async authenticateUser(credentials: AuthenticationRequest): Promise<AuthenticationToken> {
    const response = await apiClient.put('/authenticate', credentials);
    return response.data as AuthenticationToken;
  },

  async registerUser(userData: {
    username: string;
    password: string;
    permissions: string[];
  }) {
    // For Milestone 1, user registration is not implemented in the backend
    // This is a placeholder that would make an API call in a real implementation
    throw new Error('User registration not implemented in Milestone 1');
  },

  // Registry operations
  async resetRegistry() {
    const response = await apiClient.delete('/reset');
    return response.data;
  },

  // Artifact operations (OpenAPI compliant)
  async listArtifacts(queries: ArtifactQuery[], offset?: string): Promise<ArtifactMetadata[]> {
    const response = await apiClient.post('/artifacts', queries, {
      params: offset ? { offset } : {},
    });
    return response.data;
  },

  async createArtifact(artifactType: 'model' | 'dataset' | 'code', artifactData: ArtifactData): Promise<Artifact> {
    const response = await apiClient.post(`/artifact/${artifactType}`, artifactData);
    return response.data;
  },

  async getArtifact(artifactType: 'model' | 'dataset' | 'code', id: string): Promise<Artifact> {
    const response = await apiClient.get(`/artifacts/${artifactType}/${id}`);
    return response.data;
  },

  async updateArtifact(artifactType: 'model' | 'dataset' | 'code', id: string, artifact: Artifact) {
    const response = await apiClient.put(`/artifacts/${artifactType}/${id}`, artifact);
    return response.data;
  },

  async deleteArtifact(artifactType: 'model' | 'dataset' | 'code', id: string) {
    const response = await apiClient.delete(`/artifacts/${artifactType}/${id}`);
    return response.data;
  },

  // Model rating
  async rateModel(modelId: string): Promise<ModelRating> {
    const response = await apiClient.get(`/artifact/model/${modelId}/rate`);
    return response.data;
  },

  // Artifact cost
  async getArtifactCost(artifactType: 'model' | 'dataset' | 'code', id: string, includeDependencies = false): Promise<Record<string, ArtifactCost>> {
    const response = await apiClient.get(`/artifact/${artifactType}/${id}/cost`, {
      params: { dependency: includeDependencies },
    });
    return response.data;
  },

  // Tracks
  async getTracks() {
    const response = await apiClient.get('/tracks');
    return response.data;
  },

  // Legacy methods for backward compatibility (will be removed in future)
  async uploadModel(file: File, metadata: any) {
    // Convert to new artifact-based approach
    const artifactData = {
      url: URL.createObjectURL(file), // This is a placeholder - in real implementation, upload file first
    };
    return this.createArtifact('model', artifactData);
  },

  async listModels(page = 1, pageSize = 10) {
    // Convert to new artifact-based approach
    const queries: ArtifactQuery[] = [{ name: '*', types: ['model'] }];
    return this.listArtifacts(queries);
  },

  async downloadModel(modelId: string) {
    // Convert to new artifact-based approach
    return this.getArtifact('model', modelId);
  },

  async deleteModel(modelId: string) {
    // Convert to new artifact-based approach
    return this.deleteArtifact('model', modelId);
  },

  async ingestHuggingFaceModel(modelName: string) {
    // Convert to new artifact-based approach
    const artifactData = {
      url: `https://huggingface.co/${modelName}`,
    };
    return this.createArtifact('model', artifactData);
  },
};

export default apiService;
