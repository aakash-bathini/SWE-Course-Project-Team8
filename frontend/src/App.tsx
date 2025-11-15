import React, { useState, useEffect } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import {
  AppBar,
  Toolbar,
  Typography,
  Container,
  Box,
  Button,
  IconButton,
  Menu,
  MenuItem,
  Alert,
  Snackbar,
} from '@mui/material';
import {
  AccountCircle,
  CloudUpload,
  Search,
  Download,
  Dashboard,
  Security,
} from '@mui/icons-material';

// Import components
import LoginPage from './components/LoginPage';
import DashboardPage from './components/DashboardPage';
import ModelUploadPage from './components/ModelUploadPage';
import ModelSearchPage from './components/ModelSearchPage';
import ModelDownloadPage from './components/ModelDownloadPage';
import HealthDashboard from './components/HealthDashboard';
import UserManagementPage from './components/UserManagementPage';
import ModelEnumerationPage from './components/ModelEnumerationPage';
import SensitiveModelsPage from './components/SensitiveModelsPage';

// Import services
import { authService } from './services/authService';

interface User {
  username: string;
  permissions: string[];
}

const App: React.FC = () => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [notification, setNotification] = useState<{
    open: boolean;
    message: string;
    severity: 'success' | 'error' | 'warning' | 'info';
  }>({
    open: false,
    message: '',
    severity: 'info',
  });

  useEffect(() => {
    // Check if user is already logged in
    const token = localStorage.getItem('token');
    if (token) {
      // Verify token and get user info
      authService.verifyToken(token)
        .then((userData) => {
          if (userData) {
            setUser(userData);
          }
        })
        .catch((error) => {
          console.error('Token verification failed:', error);
          localStorage.removeItem('token');
        })
        .finally(() => {
          setLoading(false);
        });
    } else {
      setLoading(false);
    }
  }, []);

  const handleLogin = async (username: string, password: string) => {
    try {
      const response = await authService.login(username, password);
      if (response.token) {
        localStorage.setItem('token', response.token);
        const userData = await authService.verifyToken(response.token);
        setUser(userData);
        showNotification('Login successful!', 'success');
        return true;
      }
    } catch (error) {
      showNotification('Login failed. Please check your credentials.', 'error');
      return false;
    }
    return false;
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    setUser(null);
    setAnchorEl(null);
    showNotification('Logged out successfully', 'info');
  };

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const showNotification = (message: string, severity: 'success' | 'error' | 'warning' | 'info') => {
    setNotification({
      open: true,
      message,
      severity,
    });
  };

  const handleNotificationClose = () => {
    setNotification(prev => ({ ...prev, open: false }));
  };

  const hasPermission = (permission: string): boolean => {
    return user?.permissions.includes(permission) || false;
  };

  if (loading) {
    return (
      <Box
        display="flex"
        justifyContent="center"
        alignItems="center"
        minHeight="100vh"
      >
        <Typography>Loading...</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ flexGrow: 1 }}>
      <AppBar position="static" sx={{ backgroundColor: 'primary.main' }}>
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Trustworthy Model Registry
          </Typography>
          
          {user ? (
            <>
              <Button
                color="inherit"
                startIcon={<Dashboard />}
                href="/dashboard"
                sx={{ mr: 2 }}
              >
                Dashboard
              </Button>
              
              {hasPermission('upload') && (
                <Button
                  color="inherit"
                  startIcon={<CloudUpload />}
                  href="/upload"
                  sx={{ mr: 2 }}
                >
                  Upload
                </Button>
              )}
              
              {hasPermission('search') && (
                <Button
                  color="inherit"
                  startIcon={<Search />}
                  href="/search"
                  sx={{ mr: 2 }}
                >
                  Search
                </Button>
              )}
              
              {hasPermission('download') && (
                <Button
                  color="inherit"
                  startIcon={<Download />}
                  href="/download"
                  sx={{ mr: 2 }}
                >
                  Download
                </Button>
              )}
              
              <Button
                color="inherit"
                startIcon={<Security />}
                href="/sensitive-models"
                sx={{ mr: 2 }}
              >
                Sensitive Models
              </Button>
              
              <IconButton
                size="large"
                aria-label="account of current user"
                aria-controls="menu-appbar"
                aria-haspopup="true"
                onClick={handleMenuOpen}
                color="inherit"
              >
                <AccountCircle />
              </IconButton>
              
              <Menu
                id="menu-appbar"
                anchorEl={anchorEl}
                anchorOrigin={{
                  vertical: 'top',
                  horizontal: 'right',
                }}
                keepMounted
                transformOrigin={{
                  vertical: 'top',
                  horizontal: 'right',
                }}
                open={Boolean(anchorEl)}
                onClose={handleMenuClose}
              >
                <MenuItem onClick={handleMenuClose}>
                  <Typography>Welcome, {user.username}</Typography>
                </MenuItem>
                <MenuItem
                  onClick={() => {
                    handleMenuClose();
                    window.location.href = '/account';
                  }}
                >
                  My Account
                </MenuItem>
                <MenuItem onClick={handleLogout}>Logout</MenuItem>
              </Menu>
            </>
          ) : (
            <Button color="inherit" href="/login">
              Login
            </Button>
          )}
        </Toolbar>
      </AppBar>

      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        <Routes>
          <Route
            path="/"
            element={
              user ? <Navigate to="/dashboard" replace /> : <Navigate to="/login" replace />
            }
          />
          <Route
            path="/login"
            element={
              user ? (
                <Navigate to="/dashboard" replace />
              ) : (
                <LoginPage onLogin={handleLogin} />
              )
            }
          />
          <Route
            path="/dashboard"
            element={
              user ? (
                <DashboardPage user={user} />
              ) : (
                <Navigate to="/login" replace />
              )
            }
          />
          <Route
            path="/upload"
            element={
              user && hasPermission('upload') ? (
                <ModelUploadPage user={user} onNotification={showNotification} />
              ) : (
                <Navigate to="/login" replace />
              )
            }
          />
          <Route
            path="/search"
            element={
              user && hasPermission('search') ? (
                <ModelSearchPage user={user} />
              ) : (
                <Navigate to="/login" replace />
              )
            }
          />
          <Route
            path="/download"
            element={
              user && hasPermission('download') ? (
                <ModelDownloadPage user={user} />
              ) : (
                <Navigate to="/login" replace />
              )
            }
          />
          <Route
            path="/health"
            element={
              user ? (
                <HealthDashboard />
              ) : (
                <Navigate to="/login" replace />
              )
            }
          />
          <Route
            path="/users"
            element={
              user && hasPermission('admin') ? (
                <UserManagementPage user={user} onNotification={showNotification} />
              ) : (
                <Navigate to="/login" replace />
              )
            }
          />
          <Route
            path="/account"
            element={
              user ? (
                <UserManagementPage user={user} onNotification={showNotification} />
              ) : (
                <Navigate to="/login" replace />
              )
            }
          />
          <Route
            path="/models"
            element={
              user && hasPermission('search') ? (
                <ModelEnumerationPage user={user} />
              ) : (
                <Navigate to="/login" replace />
              )
            }
          />
          <Route
            path="/sensitive-models"
            element={
              user ? (
                <SensitiveModelsPage user={user} onNotification={showNotification} />
              ) : (
                <Navigate to="/login" replace />
              )
            }
          />
        </Routes>
      </Container>

      <Snackbar
        open={notification.open}
        autoHideDuration={6000}
        onClose={handleNotificationClose}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert
          onClose={handleNotificationClose}
          severity={notification.severity}
          sx={{ width: '100%' }}
        >
          {notification.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default App;
