import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Box,
  Card,
  CardContent,
  Grid,
  Chip,
  LinearProgress,
  Alert,
  Button,
} from '@mui/material';
import { Refresh } from '@mui/icons-material';
import { apiService } from '../services/apiService';

interface HealthData {
  status: string;
  timestamp: string;
  uptime: string;
  models_count: number;
  users_count: number;
  last_hour_activity: {
    uploads: number;
    downloads: number;
    searches: number;
  };
}

const HealthDashboard: React.FC = () => {
  const [healthData, setHealthData] = useState<HealthData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchHealthData();
  }, []);

  const fetchHealthData = async () => {
    try {
      setLoading(true);
      const data = await apiService.getHealth();
      setHealthData(data);
      setError('');
    } catch (err) {
      setError('Failed to fetch health data');
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'healthy':
        return 'success';
      case 'degraded':
        return 'warning';
      case 'unhealthy':
        return 'error';
      default:
        return 'info';
    }
  };

  if (loading) {
    return (
      <Container maxWidth="lg">
        <Box sx={{ mt: 4 }}>
          <Typography variant="h4" gutterBottom>
            System Health Dashboard
          </Typography>
          <LinearProgress />
        </Box>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg">
      <Box sx={{ mt: 4 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Typography variant="h4" gutterBottom>
            System Health Dashboard
          </Typography>
          <Button
            variant="outlined"
            startIcon={<Refresh />}
            onClick={fetchHealthData}
            disabled={loading}
          >
            Refresh
          </Button>
        </Box>

        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}

        {healthData && (
          <Grid container spacing={3}>
            {/* System Status */}
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    System Status
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <Chip
                      label={healthData.status}
                      color={getStatusColor(healthData.status) as any}
                      sx={{ mr: 2 }}
                    />
                    <Typography variant="body2" color="text.secondary">
                      Last Updated: {new Date(healthData.timestamp).toLocaleString()}
                    </Typography>
                  </Box>
                  <Typography variant="body2" color="text.secondary">
                    Uptime: {healthData.uptime}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            {/* System Metrics */}
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    System Metrics
                  </Typography>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Total Models: {healthData.models_count}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Total Users: {healthData.users_count}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            {/* Activity Metrics */}
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Last Hour Activity
                  </Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={12} sm={4}>
                      <Box sx={{ textAlign: 'center' }}>
                        <Typography variant="h4" color="primary">
                          {healthData.last_hour_activity.uploads}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Uploads
                        </Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={12} sm={4}>
                      <Box sx={{ textAlign: 'center' }}>
                        <Typography variant="h4" color="secondary">
                          {healthData.last_hour_activity.downloads}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Downloads
                        </Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={12} sm={4}>
                      <Box sx={{ textAlign: 'center' }}>
                        <Typography variant="h4" color="success.main">
                          {healthData.last_hour_activity.searches}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Searches
                        </Typography>
                      </Box>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        )}
      </Box>
    </Container>
  );
};

export default HealthDashboard;
