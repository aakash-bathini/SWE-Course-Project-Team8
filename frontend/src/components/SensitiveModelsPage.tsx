import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Box,
  TextField,
  Button,
  Stack,
  Alert,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Chip,
  IconButton,
} from '@mui/material';
import { CloudUpload, Delete, Download, Visibility, Security } from '@mui/icons-material';
import { apiService } from '../services/apiService';

interface User {
  username: string;
  permissions: string[];
}

interface SensitiveModel {
  id: string;
  model_name: string;
  uploader: string;
  js_program_id?: string;
  created_at?: string;
}

interface SensitiveModelsPageProps {
  user: User;
  onNotification: (message: string, severity: 'success' | 'error' | 'warning' | 'info') => void;
}

const SensitiveModelsPage: React.FC<SensitiveModelsPageProps> = ({ user, onNotification }) => {
  const [modelName, setModelName] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [jsProgramId, setJsProgramId] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [models, setModels] = useState<SensitiveModel[]>([]);
  const [auditResults, setAuditResults] = useState<any>(null);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [modelToDelete, setModelToDelete] = useState<string | null>(null);
  const [historyDialogOpen, setHistoryDialogOpen] = useState(false);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [downloadHistory, setDownloadHistory] = useState<any>(null);

  useEffect(() => {
    // Load package confusion audit on mount
    loadAudit();
  }, []);

  const loadAudit = async () => {
    try {
      const audit = await apiService.getPackageConfusionAudit();
      setAuditResults(audit);
    } catch (err: any) {
      console.error('Failed to load audit:', err);
    }
  };

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (!file.name.endsWith('.zip')) {
        setError('File must be a ZIP archive');
        return;
      }
      setSelectedFile(file);
      setError('');
    }
  };

  const handleUpload = async () => {
    if (!selectedFile || !modelName.trim()) {
      setError('Please select a file and enter a model name');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const result = await apiService.uploadSensitiveModel(
        selectedFile,
        modelName,
        jsProgramId || undefined
      );
      onNotification('Sensitive model uploaded successfully', 'success');
      setModelName('');
      setSelectedFile(null);
      setJsProgramId('');
      loadAudit(); // Refresh audit
    } catch (err: any) {
      const errorMsg = err.response?.data?.detail || err.message || 'Upload failed';
      setError(errorMsg);
      onNotification(`Upload failed: ${errorMsg}`, 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async () => {
    if (!modelToDelete) return;

    setLoading(true);
    try {
      await apiService.deleteSensitiveModel(modelToDelete);
      onNotification('Sensitive model deleted successfully', 'success');
      setDeleteDialogOpen(false);
      setModelToDelete(null);
      loadAudit(); // Refresh audit
    } catch (err: any) {
      const errorMsg = err.response?.data?.detail || err.message || 'Deletion failed';
      onNotification(`Deletion failed: ${errorMsg}`, 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = async (modelId: string) => {
    setLoading(true);
    try {
      const result = await apiService.downloadSensitiveModel(modelId);
      if (result.status === 'blocked') {
        onNotification(`Download blocked: ${result.message}`, 'warning');
      } else {
        onNotification('Download initiated successfully', 'success');
      }
      loadAudit(); // Refresh audit
    } catch (err: any) {
      const errorMsg = err.response?.data?.detail || err.message || 'Download failed';
      onNotification(`Download failed: ${errorMsg}`, 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleViewHistory = async (modelId: string) => {
    setSelectedModelId(modelId);
    setLoading(true);
    try {
      const history = await apiService.getDownloadHistory(modelId);
      setDownloadHistory(history);
      setHistoryDialogOpen(true);
    } catch (err: any) {
      const errorMsg = err.response?.data?.detail || err.message || 'Failed to load history';
      onNotification(`Failed to load history: ${errorMsg}`, 'error');
    } finally {
      setLoading(false);
    }
  };

  const openDeleteDialog = (modelId: string) => {
    setModelToDelete(modelId);
    setDeleteDialogOpen(true);
  };

  return (
    <Container maxWidth="lg">
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          <Security sx={{ mr: 1, verticalAlign: 'middle' }} />
          Sensitive Models Management
        </Typography>

        {/* Upload Section */}
        <Paper sx={{ p: 3, mb: 4 }}>
          <Typography variant="h6" gutterBottom>
            Upload Sensitive Model
          </Typography>
          <Stack spacing={3}>
            <TextField
              label="Model Name"
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              fullWidth
              required
            />
            <Box>
              <Button
                variant="outlined"
                component="label"
                startIcon={<CloudUpload />}
                fullWidth
              >
                Select ZIP File
                <input
                  type="file"
                  hidden
                  accept=".zip"
                  onChange={handleFileSelect}
                />
              </Button>
              {selectedFile && (
                <Typography variant="body2" sx={{ mt: 1 }}>
                  Selected: {selectedFile.name}
                </Typography>
              )}
            </Box>
            <TextField
              label="JS Program ID (Optional)"
              value={jsProgramId}
              onChange={(e) => setJsProgramId(e.target.value)}
              fullWidth
              helperText="Optional: ID of JavaScript monitoring program"
            />
            {error && <Alert severity="error">{error}</Alert>}
            <Button
              variant="contained"
              onClick={handleUpload}
              disabled={loading || !selectedFile || !modelName.trim()}
              startIcon={<CloudUpload />}
            >
              {loading ? 'Uploading...' : 'Upload Sensitive Model'}
            </Button>
          </Stack>
        </Paper>

        {/* Package Confusion Audit Section */}
        <Paper sx={{ p: 3, mb: 4 }}>
          <Typography variant="h6" gutterBottom>
            Package Confusion Audit
          </Typography>
          {auditResults && (
            <Box>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Status: {auditResults.status}
              </Typography>
              {auditResults.analysis && auditResults.analysis.length > 0 && (
                <TableContainer sx={{ mt: 2 }}>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Model ID</TableCell>
                        <TableCell>Suspicious</TableCell>
                        <TableCell>Risk Score</TableCell>
                        <TableCell>Downloads</TableCell>
                        <TableCell>Indicators</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {auditResults.analysis.map((item: any) => (
                        <TableRow key={item.model_id}>
                          <TableCell>{item.model_id}</TableCell>
                          <TableCell>
                            <Chip
                              label={item.suspicious ? 'Yes' : 'No'}
                              color={item.suspicious ? 'error' : 'success'}
                              size="small"
                            />
                          </TableCell>
                          <TableCell>{item.risk_score?.toFixed(2) || '0.00'}</TableCell>
                          <TableCell>{item.total_downloads || 0}</TableCell>
                          <TableCell>
                            {item.indicators && typeof item.indicators === 'object' ? (
                              <Box>
                                <Typography variant="caption" display="block">
                                  Bot Farm: {item.indicators.bot_farm_detected ? 'Yes' : 'No'}
                                </Typography>
                                <Typography variant="caption" display="block">
                                  Search Presence: {(item.indicators.search_presence || 0).toFixed(3)}
                                </Typography>
                              </Box>
                            ) : (
                              'N/A'
                            )}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </Box>
          )}
          <Button
            variant="outlined"
            onClick={loadAudit}
            disabled={loading}
            sx={{ mt: 2 }}
          >
            Refresh Audit
          </Button>
        </Paper>
      </Box>

      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteDialogOpen} onClose={() => setDeleteDialogOpen(false)}>
        <DialogTitle>Delete Sensitive Model</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete this sensitive model? This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleDelete} color="error" disabled={loading}>
            Delete
          </Button>
        </DialogActions>
      </Dialog>

      {/* Download History Dialog */}
      <Dialog
        open={historyDialogOpen}
        onClose={() => setHistoryDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Download History</DialogTitle>
        <DialogContent>
          {downloadHistory && (
            <Box>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Total Downloads: {downloadHistory.total_downloads || 0}
              </Typography>
              {downloadHistory.history && downloadHistory.history.length > 0 && (
                <TableContainer sx={{ mt: 2 }}>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Downloader</TableCell>
                        <TableCell>Downloaded At</TableCell>
                        <TableCell>Exit Code</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {downloadHistory.history.map((entry: any, idx: number) => (
                        <TableRow key={idx}>
                          <TableCell>{entry.downloader}</TableCell>
                          <TableCell>
                            {entry.downloaded_at
                              ? new Date(entry.downloaded_at).toLocaleString()
                              : 'N/A'}
                          </TableCell>
                          <TableCell>{entry.js_exit_code ?? 'N/A'}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setHistoryDialogOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default SensitiveModelsPage;

