import React, { useState, useEffect } from 'react';
import {
  Container,
  Paper,
  Typography,
  Box,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  CircularProgress,
  Alert,
  Pagination,
} from '@mui/material';
import { Refresh as RefreshIcon } from '@mui/icons-material';
import apiService, { ArtifactMetadata } from '../services/apiService';

interface User {
  username: string;
  permissions: string[];
}

interface ModelEnumerationPageProps {
  user: User;
}

const ModelEnumerationPage: React.FC<ModelEnumerationPageProps> = ({ user }) => {
  const [models, setModels] = useState<ArtifactMetadata[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [cursor, setCursor] = useState<string | null>(null);
  const [nextCursor, setNextCursor] = useState<string | null>(null);
  const [limit] = useState(25);

  const hasSearchPermission = user.permissions.includes('search');

  useEffect(() => {
    if (hasSearchPermission) {
      fetchModels();
    }
  }, [hasSearchPermission]);

  const fetchModels = async (nextCursorParam?: string | null) => {
    if (!hasSearchPermission) {
      setError('You do not have permission to search models');
      return;
    }

    try {
      setLoading(true);
      setError('');
      const response = await apiService.enumerateModels(nextCursorParam || null, limit);
      setModels(response.items);
      setNextCursor(response.next_cursor);
      setCursor(nextCursorParam || null);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to fetch models');
      console.error('Failed to enumerate models:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = () => {
    fetchModels(null);
  };

  const handlePageChange = (event: React.ChangeEvent<unknown>, page: number) => {
    if (page === 1) {
      fetchModels(null);
    } else {
      // For page > 1, we'd need to track cursors for each page
      // For simplicity, just fetch next page
      if (nextCursor) {
        fetchModels(nextCursor);
      }
    }
  };

  if (!hasSearchPermission) {
    return (
      <Container maxWidth="lg">
        <Alert severity="error" sx={{ mt: 4 }}>
          You do not have permission to enumerate models. 'search' permission required.
        </Alert>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg">
      <Box sx={{ mt: 4 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Typography variant="h4">Model Enumeration</Typography>
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={handleRefresh}
            disabled={loading}
          >
            Refresh
          </Button>
        </Box>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        <Paper>
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Model Name</TableCell>
                  <TableCell>ID</TableCell>
                  <TableCell>Type</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {loading ? (
                  <TableRow>
                    <TableCell colSpan={3} align="center">
                      <CircularProgress />
                    </TableCell>
                  </TableRow>
                ) : models.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={3} align="center">
                      <Typography variant="body2" color="text.secondary">
                        No models found. Use the upload or ingest endpoints to add models.
                      </Typography>
                    </TableCell>
                  </TableRow>
                ) : (
                  models.map((model) => (
                    <TableRow key={model.id}>
                      <TableCell>{model.name}</TableCell>
                      <TableCell>
                        <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                          {model.id}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Chip label={model.type} size="small" color="primary" />
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </TableContainer>

          {nextCursor && (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
              <Button
                variant="outlined"
                onClick={() => fetchModels(nextCursor)}
                disabled={loading}
              >
                Load More ({limit} more)
              </Button>
            </Box>
          )}
        </Paper>
      </Box>
    </Container>
  );
};

export default ModelEnumerationPage;

