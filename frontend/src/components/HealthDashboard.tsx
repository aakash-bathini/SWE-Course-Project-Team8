import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Box,
  Card,
  CardContent,
  Chip,
  LinearProgress,
  Alert,
  Button,
  Tabs,
  Tab,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from '@mui/material';
import { Refresh, Dashboard } from '@mui/icons-material';
import { apiService, type HealthComponent } from '../services/apiService';

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
  const [components, setComponents] = useState<HealthComponent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [tab, setTab] = useState(0);

  useEffect(() => {
    fetchHealthData();
  }, []);

  const fetchHealthData = async () => {
    try {
      setLoading(true);
      const [health, comps] = await Promise.all([
        apiService.getHealth(),
        apiService.getHealthComponents(60, false),
      ]);
      setHealthData(health);
      setComponents(comps.components);
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
      case 'ok':
        return 'success';
      case 'degraded':
        return 'warning';
      case 'unhealthy':
      case 'down':
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

        <Paper sx={{ mb: 3 }}>
          <Tabs value={tab} onChange={(_, v) => setTab(v)}>
            <Tab icon={<Dashboard />} label="Overview" />
            <Tab label="Components" />
          </Tabs>

          {tab === 0 && healthData && (
            <Box sx={{ p: 3 }}>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 3 }}>
            {/* System Status */}
            <Box sx={{ flex: '1 1 300px', minWidth: '300px' }}>
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
                          {new Date(healthData.timestamp).toLocaleString()}
                    </Typography>
                  </Box>
                  <Typography variant="body2" color="text.secondary">
                    Uptime: {healthData.uptime}
                  </Typography>
                </CardContent>
              </Card>
            </Box>

            {/* System Metrics */}
            <Box sx={{ flex: '1 1 300px', minWidth: '300px' }}>
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
            </Box>

            {/* Activity Metrics */}
            <Box sx={{ width: '100%', mt: 3 }}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Last Hour Activity
                  </Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
                    <Box sx={{ flex: '1 1 200px', textAlign: 'center' }}>
                      <Typography variant="h4" color="primary">
                        {healthData.last_hour_activity.uploads}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Uploads
                      </Typography>
                    </Box>
                    <Box sx={{ flex: '1 1 200px', textAlign: 'center' }}>
                      <Typography variant="h4" color="secondary">
                        {healthData.last_hour_activity.downloads}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Downloads
                      </Typography>
                    </Box>
                    <Box sx={{ flex: '1 1 200px', textAlign: 'center' }}>
                      <Typography variant="h4" color="success.main">
                        {healthData.last_hour_activity.searches}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Searches
                      </Typography>
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </Box>
              </Box>
            </Box>
          )}

          {tab === 1 && (
            <Box sx={{ p: 3 }}>
              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Component</TableCell>
                      <TableCell>Status</TableCell>
                      <TableCell>Description</TableCell>
                      <TableCell>Metrics</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {components.map((comp) => (
                      <TableRow key={comp.id}>
                        <TableCell>
                          <Typography variant="body2" fontWeight="medium">
                            {comp.display_name}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Chip
                            label={comp.status}
                            color={getStatusColor(comp.status) as any}
                            size="small"
                          />
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2" color="text.secondary">
                            {comp.description}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2" color="text.secondary">
                            {Object.entries(comp.metrics).map(([k, v]) => `${k}: ${v}`).join(', ')}
                          </Typography>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
          </Box>
        )}
        </Paper>
      </Box>
    </Container>
  );
};

export default HealthDashboard;
