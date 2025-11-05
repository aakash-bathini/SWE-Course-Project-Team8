import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Box,
  Card,
  CardContent,
  CardActions,
  Button,
  Chip,
  LinearProgress,
  Alert,
} from '@mui/material';
import {
  CloudUpload,
  Search,
  Download,
  Dashboard,
  Storage,
  People,
  Security,
} from '@mui/icons-material';
import apiService from '../services/apiService';

interface User {
  username: string;
  permissions: string[];
}

interface DashboardPageProps {
  user: User;
}

interface SystemStats {
  modelsCount: number;
  usersCount: number;
  lastHourActivity: {
    uploads: number;
    downloads: number;
    searches: number;
  };
}

const DashboardPage: React.FC<DashboardPageProps> = ({ user }) => {
  const [stats, setStats] = useState<SystemStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchSystemStats();
  }, []);

  const fetchSystemStats = async () => {
    try {
      const data = await apiService.getHealth();
      setStats({
        modelsCount: data.models_count,
        usersCount: data.users_count,
        lastHourActivity: data.last_hour_activity,
      });
      setError('');
    } catch (err) {
      console.error('Failed to fetch system statistics:', err);
      setError('Error connecting to the system');
    } finally {
      setLoading(false);
    }
  };

  const getPermissionChips = () => {
    return user.permissions.map((permission) => (
      <Chip
        key={permission}
        label={permission}
        color="primary"
        size="small"
        sx={{ mr: 1, mb: 1 }}
      />
    ));
  };

  const getQuickActions = () => {
    const actions = [];

    if (user.permissions.includes('upload')) {
      actions.push({
        title: 'Upload Model',
        description: 'Upload a new model to the registry',
        icon: <CloudUpload />,
        href: '/upload',
        color: 'primary',
      });
    }

    if (user.permissions.includes('search')) {
      actions.push({
        title: 'Search Models',
        description: 'Find models in the registry',
        icon: <Search />,
        href: '/search',
        color: 'secondary',
      });
    }

    if (user.permissions.includes('download')) {
      actions.push({
        title: 'Download Models',
        description: 'Download models from the registry',
        icon: <Download />,
        href: '/download',
        color: 'success',
      });
    }

    return actions;
  };

  if (loading) {
    return (
      <Container maxWidth="lg">
        <Box sx={{ mt: 4 }}>
          <Typography variant="h4" gutterBottom>
            Dashboard
          </Typography>
          <LinearProgress />
        </Box>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg">
      <Box sx={{ mt: 4 }}>
        <Typography variant="h4" gutterBottom>
          Welcome, {user.username}
        </Typography>
        <Typography variant="subtitle1" color="text.secondary" gutterBottom>
          Trustworthy Model Registry Dashboard
        </Typography>

        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}

        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 3 }}>
          {/* User Info Card */}
          <Box sx={{ flex: '1 1 300px', minWidth: '300px' }}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <People sx={{ mr: 1 }} />
                  <Typography variant="h6">User Information</Typography>
                </Box>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Username: {user.username}
                </Typography>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Permissions:
                </Typography>
                <Box sx={{ mt: 1 }}>
                  {getPermissionChips()}
                </Box>
              </CardContent>
            </Card>
          </Box>

          {/* System Stats Card */}
          <Box sx={{ flex: '1 1 300px', minWidth: '300px' }}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Dashboard sx={{ mr: 1 }} />
                  <Typography variant="h6">System Statistics</Typography>
                </Box>
                {stats ? (
                  <>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      Total Models: {stats.modelsCount}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      Total Users: {stats.usersCount}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      Last Hour Activity:
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      • Uploads: {stats.lastHourActivity.uploads}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      • Downloads: {stats.lastHourActivity.downloads}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      • Searches: {stats.lastHourActivity.searches}
                    </Typography>
                  </>
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    Unable to load statistics
                  </Typography>
                )}
              </CardContent>
            </Card>
          </Box>

          {/* Quick Actions */}
          <Box sx={{ width: '100%', mt: 3 }}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Security sx={{ mr: 1 }} />
                  <Typography variant="h6">Quick Actions</Typography>
                </Box>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
                  {getQuickActions().map((action, index) => (
                    <Box sx={{ flex: '1 1 250px', minWidth: '250px' }} key={index}>
                      <Card variant="outlined">
                        <CardContent>
                          <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                            <Box sx={{ color: `${action.color}.main`, mr: 1 }}>
                              {action.icon}
                            </Box>
                            <Typography variant="h6" component="h3">
                              {action.title}
                            </Typography>
                          </Box>
                          <Typography variant="body2" color="text.secondary">
                            {action.description}
                          </Typography>
                        </CardContent>
                        <CardActions>
                          <Button
                            size="small"
                            color={action.color as any}
                            href={action.href}
                          >
                            Go to {action.title}
                          </Button>
                        </CardActions>
                      </Card>
                    </Box>
                  ))}
                </Box>
              </CardContent>
            </Card>
          </Box>

          {/* System Health */}
          <Box sx={{ width: '100%', mt: 3 }}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Storage sx={{ mr: 1 }} />
                  <Typography variant="h6">System Health</Typography>
                </Box>
                <Button
                  variant="outlined"
                  href="/health"
                  startIcon={<Dashboard />}
                >
                  View Health Dashboard
                </Button>
              </CardContent>
            </Card>
          </Box>
        </Box>
      </Box>
    </Container>
  );
};

export default DashboardPage;
