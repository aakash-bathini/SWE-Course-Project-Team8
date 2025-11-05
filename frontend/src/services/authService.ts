import { apiService, AuthenticationRequest } from './apiService';

interface User {
  username: string;
  permissions: string[];
}

interface LoginResponse {
  token: string;
}

// Simple JWT payload decoder (Milestone 3)
// Note: We don't verify the signature here - backend validates on each request
function decodeJWT(token: string): { sub?: string; permissions?: string[] } | null {
  try {
    // Remove 'bearer ' prefix if present
    const cleanToken = token.replace(/^bearer\s+/i, '');
    
    // JWT format: header.payload.signature
    const parts = cleanToken.split('.');
    if (parts.length !== 3) {
      return null;
    }
    
    // Decode payload (base64url)
    const payload = parts[1];
    const decoded = atob(payload.replace(/-/g, '+').replace(/_/g, '/'));
    return JSON.parse(decoded);
  } catch (error) {
    console.error('JWT decode failed:', error);
    return null;
  }
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
    // Store raw string token (includes 'bearer ' prefix)
    localStorage.setItem('token', tokenString);
    return { token: tokenString };
  },

  async verifyToken(token: string): Promise<User | null> {
    try {
      // Milestone 3: Decode JWT token to extract user info
      if (typeof token === 'string' && token.trim().length > 0) {
        const decoded = decodeJWT(token);
        if (decoded && decoded.sub) {
          return {
            username: decoded.sub,
            permissions: decoded.permissions || [],
          };
        }
        // Fallback for invalid tokens (shouldn't happen, but handle gracefully)
        return null;
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
    // Milestone 3: Admin-only user registration
    const response = await apiService.registerUser(userData);
    return response;
  },

  async deleteUser(username: string): Promise<any> {
    // Milestone 3: User deletion (users can delete own, admins can delete any)
    const response = await apiService.deleteUser(username);
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
