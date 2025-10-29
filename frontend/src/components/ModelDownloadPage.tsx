import React, { useEffect, useState } from 'react';
import {
  Container,
  Typography,
  Box,
  Stack,
  Button,
  TextField,
  MenuItem,
  Alert,
  Paper,
  List,
  ListItem,
  ListItemText,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
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
  const [info, setInfo] = useState('');
  const [licenseDialog, setLicenseDialog] = useState<{ open: boolean; id?: string }>({ open: false });
  const [licenseUrl, setLicenseUrl] = useState('https://github.com/google-research/bert');

  const canDownload = user.permissions.includes('download') || user.permissions.includes('search');

  const fetchList = async () => {
    setLoading(true);
    setError('');
    try {
      const queries = [{ name: nameQuery, types: [artifactType] }];
      const data = await apiService.listArtifacts(queries, '0');
      setItems(data);
      setInfo(`Found ${data.length} item(s).`);
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
        {info && !error && <Alert severity="info" sx={{ mb: 2 }}>{info}</Alert>}
        <Paper variant="outlined">
          <List dense>
            {items.map((m) => (
              <ListItem key={m.id} secondaryAction={
                <Stack direction="row" spacing={1}>
                  {m.type === 'model' && (
                    <>
                      <Button size="small" onClick={async () => {
                        try {
                          const rating = await apiService.rateModel(m.id);
                          setInfo(`Rating for ${m.name}: net_score=${rating.net_score.toFixed(2)}`);
                          setError('');
                        } catch (e: any) {
                          setError(e?.message || 'Failed to fetch rating');
                        }
                      }}>Rate</Button>
                      <Button size="small" onClick={async () => {
                        try {
                          const lineage = await apiService.getModelLineage(m.id);
                          setInfo(`Lineage nodes=${(lineage?.nodes?.length||0)} edges=${(lineage?.edges?.length||0)}`);
                          setError('');
                        } catch (e: any) {
                          setError(e?.message || 'Failed to fetch lineage');
                        }
                      }}>Lineage</Button>
                      <Button size="small" onClick={() => setLicenseDialog({ open: true, id: m.id })}>License Check</Button>
                    </>
                  )}
                  <Button size="small" onClick={async () => {
                    try {
                      const cost = await apiService.getArtifactCost(m.type, m.id, false);
                      setInfo(`Cost (MB) for ${m.name}: ${cost[m.id]?.total_cost}`);
                      setError('');
                    } catch (e: any) {
                      setError(e?.message || 'Failed to fetch cost');
                    }
                  }}>Cost</Button>
                  <Button size="small" onClick={async () => {
                    try {
                      const cost = await apiService.getArtifactCost(m.type, m.id, true);
                      const standalone = cost[m.id]?.standalone_cost ?? 'n/a';
                      setInfo(`Cost+Deps (MB) for ${m.name}: total=${cost[m.id]?.total_cost}, standalone=${standalone}`);
                      setError('');
                    } catch (e: any) {
                      setError(e?.message || 'Failed to fetch dependency cost');
                    }
                  }}>Cost+Deps</Button>
                </Stack>
              }>
                <ListItemText primary={`${m.name} (${m.type})`} secondary={`id: ${m.id}`} />
              </ListItem>
            ))}
            {!items.length && <ListItem><ListItemText primary="No artifacts found" /></ListItem>}
          </List>
        </Paper>
      </Box>

      <Dialog open={licenseDialog.open} onClose={() => setLicenseDialog({ open: false })} fullWidth maxWidth="sm">
        <DialogTitle>License Check</DialogTitle>
        <DialogContent>
          <TextField
            fullWidth
            label="GitHub Repository URL"
            value={licenseUrl}
            onChange={(e) => setLicenseUrl(e.target.value)}
            sx={{ mt: 1 }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setLicenseDialog({ open: false })}>Cancel</Button>
          <Button
            variant="contained"
            onClick={async () => {
              if (!licenseDialog.id) return;
              try {
                const ok = await apiService.licenseCheck(licenseDialog.id, licenseUrl);
                setInfo(`License compatibility: ${ok ? 'compatible' : 'incompatible'}`);
                setError('');
              } catch (e: any) {
                setError(e?.message || 'License check failed');
              } finally {
                setLicenseDialog({ open: false });
              }
            }}
          >Run Check</Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default ModelDownloadPage;
