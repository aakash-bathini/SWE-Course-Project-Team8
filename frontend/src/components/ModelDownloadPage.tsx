import React, { useEffect, useState } from 'react';
import { Container, Typography, Box, Stack, Button, TextField, MenuItem, Alert, Paper, List, ListItem, ListItemText } from '@mui/material';
import { apiService, type ArtifactMetadata } from '../services/apiService';

interface User {
  username: string;
  permissions: string[];
}

interface ModelDownloadPageProps {
  user: User;
}

const ModelDownloadPage: React.FC<ModelDownloadPageProps> = ({ user }) => {
  const [artifactType, setArtifactType] = useState<'model' | 'dataset' | 'code'>('model');
  const [nameQuery, setNameQuery] = useState('*');
  const [items, setItems] = useState<ArtifactMetadata[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const canDownload = user.permissions.includes('download') || user.permissions.includes('search');

  const fetchList = async () => {
    setLoading(true);
    setError('');
    try {
      const queries = [{ name: nameQuery, types: [artifactType] }];
      const data = await apiService.listArtifacts(queries, '0');
      setItems(data);
    } catch (e: any) {
      setError(e?.message || 'Failed to list artifacts');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchList();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <Container maxWidth="lg">
      <Box sx={{ mt: 4 }}>
        <Typography variant="h4" gutterBottom>
          Download Models
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          User: {user.username} | Permissions: {user.permissions.join(', ')}
        </Typography>
        {!canDownload && (
          <Alert severity="warning" sx={{ mb: 2 }}>
            Your account does not have 'search' or 'download' permissions.
          </Alert>
        )}
        <Stack spacing={2} direction={{ xs: 'column', sm: 'row' }} sx={{ mb: 2 }}>
          <TextField
            label="Artifact Type"
            select
            value={artifactType}
            onChange={(e) => setArtifactType(e.target.value as any)}
            sx={{ minWidth: 180 }}
          >
            <MenuItem value="model">model</MenuItem>
            <MenuItem value="dataset">dataset</MenuItem>
            <MenuItem value="code">code</MenuItem>
          </TextField>
          <TextField label="Name" value={nameQuery} onChange={(e) => setNameQuery(e.target.value)} fullWidth />
          <Button variant="outlined" onClick={fetchList} disabled={loading || !canDownload}>
            {loading ? 'Loading...' : 'Search'}
          </Button>
        </Stack>
        {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
        <Paper variant="outlined">
          <List dense>
            {items.map((m) => (
              <ListItem key={m.id} secondaryAction={
                <Stack direction="row" spacing={1}>
                  <Button size="small" href={`/api/docs#/${artifactType}/get_artifacts_${artifactType}__id_`} target="_blank">Docs</Button>
                </Stack>
              }>
                <ListItemText primary={`${m.name} (${m.type})`} secondary={`id: ${m.id}`} />
              </ListItem>
            ))}
            {!items.length && <ListItem><ListItemText primary="No artifacts found" /></ListItem>}
          </List>
        </Paper>
      </Box>
    </Container>
  );
};

export default ModelDownloadPage;
