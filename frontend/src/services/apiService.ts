import axios, { type AxiosRequestHeaders } from 'axios';

// API URL configuration: Use environment variable for production, default to localhost for development
// In production (AWS Amplify), set REACT_APP_API_URL to your API Gateway URL
// Example: REACT_APP_API_URL=https://han6e7iv6e.execute-api.us-east-1.amazonaws.com
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
    // Avoid attaching auth headers to public health endpoint to prevent CORS preflight
    try {
      const urlStr = (config.url ?? '').toString();
      if (urlStr.endsWith('/health')) {
        return config;
      }
    } catch (e) {
      // no-op safeguard
    }
    const token = localStorage.getItem('token');
    if (token) {
      // Ensure headers object exists and is typed for axios
      const headers: AxiosRequestHeaders = (config.headers as AxiosRequestHeaders) || {} as AxiosRequestHeaders;
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
  download_url?: string;
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

export interface HealthComponent {
  id: string;
  display_name: string;
  status: 'ok' | 'degraded' | 'down';
  observed_at: string;
  description: string;
  metrics: Record<string, any>;
  issues: string[];
  logs: Array<{ label: string; url: string }>;
  timeline?: Array<{ bucket: string; value: number; unit: string }>;
}

export interface HealthComponentsResponse {
  components: HealthComponent[];
  generated_at: string;
  window_minutes: number;
}

export interface ArtifactAuditEntry {
  user: {
    name: string;
    is_admin: boolean;
  };
  date: string;
  artifact: ArtifactMetadata;
  action: string;
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

// User registration request (Milestone 3)
export interface UserRegistrationRequest {
  username: string;
  password: string;
  permissions: string[];
}

// Models enumeration response (Milestone 2)
export interface ModelsEnumerateResponse {
  items: ArtifactMetadata[];
  next_cursor: string | null;
}

export const apiService = {
  // Health check
  async getHealth() {
    const response = await apiClient.get('/health');
    return response.data;
  },

  async getHealthComponents(windowMinutes = 60, includeTimeline = false) {
    const response = await apiClient.get('/health/components', {
      params: { windowMinutes, includeTimeline },
    });
    return response.data as HealthComponentsResponse;
  },

  // Authentication
  async authenticateUser(credentials: AuthenticationRequest): Promise<AuthenticationToken> {
    const response = await apiClient.put('/authenticate', credentials);
    return response.data as AuthenticationToken;
  },

  async registerUser(userData: UserRegistrationRequest) {
    // Milestone 3: Admin-only user registration endpoint
    const response = await apiClient.post('/register', userData);
    return response.data;
  },

  // User management (Milestone 3)
  async deleteUser(username: string) {
    const response = await apiClient.delete(`/user/${encodeURIComponent(username)}`);
    return response.data;
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

  async searchByName(name: string): Promise<ArtifactMetadata[]> {
    if (name === '*') {
      const response = await apiClient.post('/artifacts', [{ name: '*' }], { params: { offset: 0 } });
      return response.data;
    }
    const response = await apiClient.get(`/artifact/byName/${encodeURIComponent(name)}`);
    return response.data;
  },

  async searchByRegex(regex: string): Promise<ArtifactMetadata[]> {
    const response = await apiClient.post('/artifact/byRegEx', { regex });
    return response.data;
  },

  async createArtifact(artifactType: 'model' | 'dataset' | 'code', artifactData: ArtifactData): Promise<Artifact> {
    // Bump timeout for URL registration path to better tolerate slower networks
    const response = await apiClient.post(`/artifact/${artifactType}`, artifactData, { timeout: 30000 });
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

  // Model lineage (graph)
  async getModelLineage(modelId: string) {
    const response = await apiClient.get(`/artifact/model/${modelId}/lineage`);
    return response.data;
  },

  // License check
  async licenseCheck(modelId: string, githubUrl: string): Promise<boolean> {
    const response = await apiClient.post(`/artifact/model/${modelId}/license-check`, {
      github_url: githubUrl,
    });
    return response.data as boolean;
  },

  // Artifact cost
  async getArtifactCost(artifactType: 'model' | 'dataset' | 'code', id: string, includeDependencies = false): Promise<Record<string, ArtifactCost>> {
    const response = await apiClient.get(`/artifact/${artifactType}/${id}/cost`, {
      params: { dependency: includeDependencies },
    });
    return response.data;
  },

  // Artifact audit trail
  async getArtifactAudit(artifactType: 'model' | 'dataset' | 'code', id: string): Promise<ArtifactAuditEntry[]> {
    const response = await apiClient.get(`/artifact/${artifactType}/${id}/audit`);
    return response.data as ArtifactAuditEntry[];
  },

  // Users (Milestone 3)
  async getUsers(): Promise<Array<{ username: string; permissions: string[] }>> {
    const response = await apiClient.get('/users');
    return response.data as Array<{ username: string; permissions: string[] }>;
  },

  // Model file download (ZIP)
  async downloadModel(id: string, aspect: 'full' | 'weights' | 'datasets' | 'code' = 'full'): Promise<Blob> {
    const response = await apiClient.get(`/models/${id}/download`, {
      params: { aspect },
      responseType: 'blob',
    });
    return response.data as Blob;
  },

  // Model ZIP upload
  async uploadModelZip(file: File, name?: string): Promise<Artifact> {
    const formData = new FormData();
    formData.append('file', file);
    if (name) {
      formData.append('name', name);
    }
    const response = await apiClient.post('/models/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data as Artifact;
  },

  // Model ingestion (Milestone 2)
  async ingestHuggingFaceModel(modelName: string): Promise<Artifact> {
    const response = await apiClient.post('/models/ingest', null, {
      params: { model_name: modelName },
    });
    return response.data as Artifact;
  },

  // Model enumeration (Milestone 2)
  async enumerateModels(cursor?: string | null, limit: number = 25): Promise<ModelsEnumerateResponse> {
    const params: Record<string, string | number> = { limit };
    if (cursor) {
      params.cursor = cursor;
    }
    const response = await apiClient.get('/models', { params });
    return response.data as ModelsEnumerateResponse;
  },

  // Tracks
  async getTracks() {
    const response = await apiClient.get('/tracks');
    return response.data;
  },
};

export default apiService;
