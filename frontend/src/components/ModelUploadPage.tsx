import React, { useState } from 'react';
import { Container, Typography, Box, TextField, Button, MenuItem, Stack, Alert } from '@mui/material';
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
  const [artifactType, setArtifactType] = useState<'model' | 'dataset' | 'code'>('model');
  const [url, setUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const canUpload = user.permissions.includes('upload');

  const handleUpload = async () => {
    if (!canUpload) {
      onNotification('You do not have permission to upload artifacts.', 'error');
      return;
    }
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
      onNotification(e?.message || 'Upload failed', 'error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="lg">
      <Box sx={{ mt: 4 }}>
        <Typography variant="h4" gutterBottom>
          Upload Model
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          User: {user.username} | Permissions: {user.permissions.join(', ')}
        </Typography>
        {!canUpload && (
          <Alert severity="warning" sx={{ mb: 2 }}>
            Your account does not have the 'upload' permission.
          </Alert>
        )}
        <Stack spacing={2} direction={{ xs: 'column', sm: 'row' }}>
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
            label="Source URL (e.g., https://huggingface.co/org/model)"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            disabled={loading}
            error={Boolean(error)}
            helperText={error || 'Provide a single downloadable source URL'}
          />
          <Button variant="contained" onClick={handleUpload} disabled={loading || !canUpload}>
            {loading ? 'Registering...' : 'Register Artifact'}
          </Button>
        </Stack>
      </Box>
    </Container>
  );
};

export default ModelUploadPage;
