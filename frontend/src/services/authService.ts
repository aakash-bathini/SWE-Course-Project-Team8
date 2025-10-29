import { apiService, AuthenticationRequest } from './apiService';

interface User {
  username: string;
  permissions: string[];
}

interface LoginResponse {
  token: string;
}

export const authService = {
  async login(username: string, password: string, isAdmin = false): Promise<LoginResponse> {
    const credentials: AuthenticationRequest = {
      user: {
        name: username,
        is_admin: isAdmin,
      },
      secret: {
        password: password,
      },
    };
    
    const tokenString = await apiService.authenticateUser(credentials);
    // Store raw string token
    localStorage.setItem('token', tokenString);
    return { token: tokenString };
  },

  async verifyToken(token: string): Promise<User | null> {
    try {
      // Delivery 1 behavior: backend returns a raw string token (already includes 'bearer ')
      // Accept any non-empty string as authenticated and provide default admin permissions.
      // Future Milestone: replace with actual verify endpoint and claims parsing.
      if (typeof token === 'string' && token.trim().length > 0) {
        return {
          username: 'ece30861defaultadminuser',
          permissions: ['upload', 'search', 'download', 'admin'],
        };
      }
      return null;
    } catch (error) {
      console.error('Token verification failed:', error);
      return null;
    }
  },

  async registerUser(userData: {
    username: string;
    password: string;
    permissions: string[];
  }): Promise<any> {
    const response = await apiService.registerUser(userData);
    return response;
  },

  logout(): void {
    localStorage.removeItem('token');
  },

  isAuthenticated(): boolean {
    const token = localStorage.getItem('token');
    return !!token;
  },

  getToken(): string | null {
    return localStorage.getItem('token');
  },
};

export default authService;
