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
      // For Milestone 1, we'll use a simplified token verification
      // In Milestone 3, this will make an actual API call to verify the JWT
      if (token === 'demo_token') {
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
