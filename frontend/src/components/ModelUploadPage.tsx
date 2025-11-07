import React, { useState } from 'react';
import {
  Container,
  Typography,
  Box,
  TextField,
  Button,
  MenuItem,
  Stack,
  Alert,
  Paper,
  Tabs,
  Tab,
  Input,
  FormControlLabel,
  Switch,
} from '@mui/material';
import { CloudUpload, Link as LinkIcon, UploadFile } from '@mui/icons-material';
import { apiService } from '../services/apiService';

interface User {
  username: string;
  permissions: string[];
}

interface ModelUploadPageProps {
  user: User;
  onNotification: (message: string, severity: 'success' | 'error' | 'warning' | 'info') => void;
}

const ModelUploadPage: React.FC<ModelUploadPageProps> = ({ user, onNotification }) => {
  const [uploadMode, setUploadMode] = useState<'url' | 'zip' | 'huggingface'>('url');
  const [artifactType, setArtifactType] = useState<'model' | 'dataset' | 'code'>('model');
  const [url, setUrl] = useState('');
  const [modelName, setModelName] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [hfModelName, setHfModelName] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const canUpload = user.permissions.includes('upload');

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (!file.name.endsWith('.zip')) {
        setError('File must be a ZIP archive');
      return;
    }
      setSelectedFile(file);
      setError('');
      // Auto-set model name from filename
      if (!modelName) {
        setModelName(file.name.replace('.zip', ''));
      }
    }
  };

  const handleUrlUpload = async () => {
    if (!url.trim()) {
      setError('Please provide a valid source URL.');
      return;
    }
    setError('');
    setLoading(true);
    try {
      const result = await apiService.createArtifact(artifactType, { url: url.trim() });
      onNotification(`Registered ${result.metadata.type} '${result.metadata.name}' (id: ${result.metadata.id})`, 'success');
      setUrl('');
    } catch (e: any) {
      handleError(e);
    } finally {
      setLoading(false);
    }
  };

  const handleZipUpload = async () => {
    if (!selectedFile) {
      setError('Please select a ZIP file to upload.');
      return;
    }
    setError('');
    setLoading(true);
    try {
      const result = await apiService.uploadModelZip(selectedFile, modelName || undefined);
      onNotification(`Uploaded model '${result.metadata.name}' (id: ${result.metadata.id})`, 'success');
      setSelectedFile(null);
      setModelName('');
      // Reset file input
      const fileInput = document.getElementById('zip-file-input') as HTMLInputElement;
      if (fileInput) fileInput.value = '';
    } catch (e: any) {
      handleError(e);
    } finally {
      setLoading(false);
    }
  };

  const handleHuggingFaceUpload = async () => {
    if (!hfModelName.trim()) {
      setError('Please provide a HuggingFace model name (e.g., google/gemma-2-2b).');
      return;
    }
    setError('');
    setLoading(true);
    try {
      const result = await apiService.ingestHuggingFaceModel(hfModelName.trim());
      onNotification(`Ingested HuggingFace model '${result.metadata.name}' (id: ${result.metadata.id})`, 'success');
      setHfModelName('');
    } catch (e: any) {
      handleError(e);
    } finally {
      setLoading(false);
    }
  };

  const handleError = (e: any) => {
    const status = e?.response?.status;
    if (status === 424) {
      onNotification('Upload rejected: model does not meet quality threshold (HTTP 424).', 'warning');
    } else if (status === 403 || status === 401) {
      onNotification('Authentication required. Please sign in again.', 'error');
    } else if (status === 400) {
      onNotification(e?.response?.data?.detail || 'Invalid request. Please check your input.', 'error');
    } else {
      onNotification(e?.response?.data?.detail || e?.message || 'Upload failed', 'error');
    }
  };

  return (
    <Container maxWidth="lg">
      <Box sx={{ mt: 4 }}>
        <Typography variant="h4" gutterBottom>
          Upload Artifact
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          User: {user.username} | Permissions: {user.permissions.join(', ')}
        </Typography>

        {!canUpload && (
          <Alert severity="warning" sx={{ mb: 2 }}>
            Your account does not have the 'upload' permission.
          </Alert>
        )}

        <Paper sx={{ mb: 3 }}>
          <Tabs value={uploadMode} onChange={(_, v) => setUploadMode(v)} sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tab icon={<LinkIcon />} label="URL" value="url" />
            <Tab icon={<UploadFile />} label="ZIP File" value="zip" />
            <Tab icon={<CloudUpload />} label="HuggingFace" value="huggingface" />
          </Tabs>

          <Box sx={{ p: 3 }}>
            {uploadMode === 'url' && (
              <Stack spacing={2}>
          <TextField
            label="Artifact Type"
            select
            value={artifactType}
            onChange={(e) => setArtifactType(e.target.value as any)}
            sx={{ minWidth: 180 }}
            disabled={loading}
          >
            <MenuItem value="model">model</MenuItem>
            <MenuItem value="dataset">dataset</MenuItem>
            <MenuItem value="code">code</MenuItem>
          </TextField>
          <TextField
            fullWidth
                  label="Source URL"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            disabled={loading}
            error={Boolean(error)}
                  helperText={error || 'Provide a source URL (e.g., https://example.com/resource)'}
                  placeholder="https://example.com/resource"
          />
                <Button variant="contained" onClick={handleUrlUpload} disabled={loading || !canUpload}>
            {loading ? 'Registering...' : 'Register Artifact'}
          </Button>
        </Stack>
            )}

            {uploadMode === 'zip' && (
              <Stack spacing={2}>
                <Alert severity="info" sx={{ mb: 2 }}>
                  Upload a ZIP file containing model files. The ZIP should include a README.md file with model card information.
                </Alert>
                <TextField
                  label="Model Name (optional)"
                  value={modelName}
                  onChange={(e) => setModelName(e.target.value)}
                  disabled={loading}
                  helperText="If not provided, filename will be used"
                  fullWidth
                />
                <Box>
                  <Input
                    id="zip-file-input"
                    type="file"
                    inputProps={{ accept: '.zip' }}
                    onChange={handleFileSelect}
                    disabled={loading}
                    sx={{ display: 'none' }}
                  />
                  <label htmlFor="zip-file-input">
                    <Button variant="outlined" component="span" disabled={loading} startIcon={<UploadFile />}>
                      {selectedFile ? selectedFile.name : 'Select ZIP File'}
                    </Button>
                  </label>
                  {selectedFile && (
                    <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                      Selected: {selectedFile.name} ({(selectedFile.size / 1024 / 1024).toFixed(2)} MB)
                    </Typography>
                  )}
                </Box>
                {error && <Alert severity="error">{error}</Alert>}
                <Button variant="contained" onClick={handleZipUpload} disabled={loading || !canUpload || !selectedFile}>
                  {loading ? 'Uploading...' : 'Upload ZIP'}
                </Button>
              </Stack>
            )}

            {uploadMode === 'huggingface' && (
              <Stack spacing={2}>
                <Alert severity="info" sx={{ mb: 2 }}>
                  Ingest a model from HuggingFace. The model must meet quality thresholds (all non-latency metrics ≥ 0.5).
                </Alert>
                <TextField
                  fullWidth
                  label="HuggingFace Model Name"
                  value={hfModelName}
                  onChange={(e) => setHfModelName(e.target.value)}
                  disabled={loading}
                  error={Boolean(error)}
                  helperText={error || 'e.g., google/gemma-2-2b or microsoft/DialoGPT-medium'}
                  placeholder="google/gemma-2-2b"
                />
                {error && <Alert severity="error">{error}</Alert>}
                <Button variant="contained" onClick={handleHuggingFaceUpload} disabled={loading || !canUpload}>
                  {loading ? 'Ingesting...' : 'Ingest from HuggingFace'}
                </Button>
              </Stack>
            )}
          </Box>
        </Paper>
      </Box>
    </Container>
  );
};

export default ModelUploadPage;
