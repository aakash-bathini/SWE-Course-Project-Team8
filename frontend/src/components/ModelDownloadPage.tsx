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
  Chip,
  IconButton,
  Tooltip,
  Menu,
} from '@mui/material';
import {
  Download,
  History,
} from '@mui/icons-material';
import { apiService, type ArtifactMetadata, type ArtifactAuditEntry, type ModelRating } from '../services/apiService';

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
  const [auditDialog, setAuditDialog] = useState<{ open: boolean; artifact?: ArtifactMetadata }>({ open: false });
  const [auditEntries, setAuditEntries] = useState<ArtifactAuditEntry[]>([]);
  const [ratingDialog, setRatingDialog] = useState<{ open: boolean; rating?: ModelRating; artifactName?: string }>({ open: false });
  const [downloadAnchorEl, setDownloadAnchorEl] = useState<null | HTMLElement>(null);
  const [selectedArtifactForDownload, setSelectedArtifactForDownload] = useState<ArtifactMetadata | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);

  const canDownload = user.permissions.includes('download') || user.permissions.includes('search');

  const fetchList = async () => {
    setLoading(true);
    setError('');
    try {
      const qName = (nameQuery || '').trim();
      const queries = qName === '*' || !qName
        ? [{ name: '*', types: [artifactType] } as any]
        : [{ name: qName, types: [artifactType] } as any];
      const data = await apiService.listArtifacts(queries, '0');
      setItems(data);
      setInfo(`Found ${data.length} item(s).`);
    } catch (e: any) {
      setError(e?.response?.data?.detail || e?.message || 'Failed to list artifacts');
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = async (artifact: ArtifactMetadata, aspect: 'full' | 'weights' | 'datasets' | 'code' = 'full') => {
    if (artifact.type !== 'model') {
      setError('File download is only available for models.');
      return;
    }
    setBusyId(artifact.id);
    try {
      const blob = await apiService.downloadModel(artifact.id, aspect);
      // Create download link
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${artifact.name}_${aspect}.zip`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      setInfo(`Downloaded ${artifact.name} (${aspect})`);
      setError('');
    } catch (e: any) {
      setError(e?.response?.data?.detail || e?.message || 'Download failed');
    } finally {
      setBusyId(null);
    }
  };

  const handleViewAudit = async (artifact: ArtifactMetadata) => {
    setAuditDialog({ open: true, artifact });
    setBusyId(artifact.id);
    try {
      const entries = await apiService.getArtifactAudit(artifact.type, artifact.id);
      setAuditEntries(entries);
    } catch (e: any) {
      setError(e?.response?.data?.detail || e?.message || 'Failed to fetch audit trail');
    } finally {
      setBusyId(null);
    }
  };

  useEffect(() => {
    fetchList();
  }, []); // Empty deps array is intentional - only run on mount

  return (
    <Container maxWidth="lg">
      <Box sx={{ mt: 4 }}>
        <Typography variant="h4" gutterBottom>
          Browse & Download Artifacts
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
          <TextField label="Name (use * for all)" value={nameQuery} onChange={(e) => setNameQuery(e.target.value)} fullWidth />
          <Button variant="outlined" onClick={fetchList} disabled={loading || !canDownload}>
            {loading ? 'Loading...' : 'Search'}
          </Button>
        </Stack>
        {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
        {info && !error && <Alert severity="info" sx={{ mb: 2 }}>{info}</Alert>}
        <Paper variant="outlined">
          <List dense>
            {items.map((m) => (
              <ListItem
                key={m.id}
                secondaryAction={
                  <Stack direction="row" spacing={1} flexWrap="wrap">
                    {m.type === 'model' && (
                      <>
                        <Tooltip title="Download model">
                          <IconButton
                            size="small"
                            disabled={!!busyId}
                            onClick={(e) => {
                              setSelectedArtifactForDownload(m);
                              setDownloadAnchorEl(e.currentTarget);
                            }}
                            color="primary"
                          >
                            <Download />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="View audit trail">
                          <IconButton
                            size="small"
                            disabled={!!busyId}
                            onClick={() => handleViewAudit(m)}
                            color="secondary"
                          >
                            <History />
                          </IconButton>
                        </Tooltip>
                        <Button size="small" disabled={!!busyId} onClick={async () => {
                          try {
                            setBusyId(m.id);
                            const rating = await apiService.rateModel(m.id);
                            setRatingDialog({ open: true, rating, artifactName: m.name });
                            setError('');
                          } catch (e: any) {
                            setError(e?.response?.data?.detail || e?.message || 'Failed to fetch rating');
                          } finally { setBusyId(null); }
                        }}>Rate</Button>
                        <Button size="small" disabled={!!busyId} onClick={async () => {
                          try {
                            setBusyId(m.id);
                            const lineage = await apiService.getModelLineage(m.id);
                            setInfo(`Lineage: ${lineage?.nodes?.length || 0} nodes, ${lineage?.edges?.length || 0} edges`);
                            setError('');
                          } catch (e: any) {
                            setError(e?.response?.data?.detail || e?.message || 'Failed to fetch lineage');
                          } finally { setBusyId(null); }
                        }}>Lineage</Button>
                        <Button size="small" disabled={!!busyId} onClick={() => setLicenseDialog({ open: true, id: m.id })}>License</Button>
                      </>
                    )}
                    <Button size="small" disabled={!!busyId} onClick={async () => {
                      try {
                        setBusyId(m.id);
                        const cost = await apiService.getArtifactCost(m.type, m.id, false);
                        setInfo(`Cost (MB) for ${m.name}: ${cost[m.id]?.total_cost}`);
                        setError('');
                      } catch (e: any) {
                        setError(e?.response?.data?.detail || e?.message || 'Failed to fetch cost');
                      } finally { setBusyId(null); }
                    }}>Cost</Button>
                  </Stack>
                }
              >
                <ListItemText
                  primary={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Typography variant="body1">{m.name}</Typography>
                      <Chip label={m.type} size="small" color="primary" />
                    </Box>
                  }
                  secondary={`ID: ${m.id}`}
                />
              </ListItem>
            ))}
            {!items.length && <ListItem><ListItemText primary="No artifacts found" /></ListItem>}
          </List>
        </Paper>
      </Box>

      {/* License Check Dialog */}
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
                setError(e?.response?.data?.detail || e?.message || 'License check failed');
              } finally {
                setLicenseDialog({ open: false });
              }
            }}
          >Run Check</Button>
        </DialogActions>
      </Dialog>

      {/* Download Aspect Menu */}
      <Menu
        anchorEl={downloadAnchorEl}
        open={Boolean(downloadAnchorEl)}
        onClose={() => {
          setDownloadAnchorEl(null);
          setSelectedArtifactForDownload(null);
        }}
      >
        <MenuItem onClick={() => {
          if (selectedArtifactForDownload) {
            handleDownload(selectedArtifactForDownload, 'full');
          }
          setDownloadAnchorEl(null);
          setSelectedArtifactForDownload(null);
        }}>
          Download Full Model
        </MenuItem>
        <MenuItem onClick={() => {
          if (selectedArtifactForDownload) {
            handleDownload(selectedArtifactForDownload, 'weights');
          }
          setDownloadAnchorEl(null);
          setSelectedArtifactForDownload(null);
        }}>
          Download Weights Only
        </MenuItem>
        <MenuItem onClick={() => {
          if (selectedArtifactForDownload) {
            handleDownload(selectedArtifactForDownload, 'datasets');
          }
          setDownloadAnchorEl(null);
          setSelectedArtifactForDownload(null);
        }}>
          Download Datasets Only
        </MenuItem>
        <MenuItem onClick={() => {
          if (selectedArtifactForDownload) {
            handleDownload(selectedArtifactForDownload, 'code');
          }
          setDownloadAnchorEl(null);
          setSelectedArtifactForDownload(null);
        }}>
          Download Code Only
        </MenuItem>
      </Menu>

      {/* Rating Details Dialog */}
      <Dialog open={ratingDialog.open} onClose={() => setRatingDialog({ open: false })} fullWidth maxWidth="md">
        <DialogTitle>
          Model Rating: {ratingDialog.artifactName}
        </DialogTitle>
        <DialogContent>
          {ratingDialog.rating ? (
            <Box sx={{ mt: 2 }}>
              <Typography variant="h6" gutterBottom>
                Overall Score: {ratingDialog.rating.net_score.toFixed(3)}
              </Typography>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Category: {ratingDialog.rating.category}
              </Typography>
              <Typography variant="subtitle2" sx={{ mt: 2, mb: 1 }}>
                Phase 1 Metrics:
              </Typography>
              <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 1 }}>
                <Typography variant="body2">Ramp-up Time: {ratingDialog.rating.ramp_up_time.toFixed(3)}</Typography>
                <Typography variant="body2">Bus Factor: {ratingDialog.rating.bus_factor.toFixed(3)}</Typography>
                <Typography variant="body2">Performance Claims: {ratingDialog.rating.performance_claims.toFixed(3)}</Typography>
                <Typography variant="body2">License: {ratingDialog.rating.license.toFixed(3)}</Typography>
                <Typography variant="body2">Dataset & Code Score: {ratingDialog.rating.dataset_and_code_score.toFixed(3)}</Typography>
                <Typography variant="body2">Dataset Quality: {ratingDialog.rating.dataset_quality.toFixed(3)}</Typography>
                <Typography variant="body2">Code Quality: {ratingDialog.rating.code_quality.toFixed(3)}</Typography>
              </Box>
              <Typography variant="subtitle2" sx={{ mt: 2, mb: 1 }}>
                Phase 2 Metrics:
              </Typography>
              <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 1 }}>
                <Typography variant="body2">Reproducibility: {ratingDialog.rating.reproducibility.toFixed(3)}</Typography>
                <Typography variant="body2">Reviewedness: {ratingDialog.rating.reviewedness.toFixed(3)}</Typography>
                <Typography variant="body2">Tree Score: {ratingDialog.rating.tree_score.toFixed(3)}</Typography>
              </Box>
              <Typography variant="subtitle2" sx={{ mt: 2, mb: 1 }}>
                Size Scores:
              </Typography>
              <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 1 }}>
                <Typography variant="body2">Raspberry Pi: {ratingDialog.rating.size_score.raspberry_pi.toFixed(3)}</Typography>
                <Typography variant="body2">Jetson Nano: {ratingDialog.rating.size_score.jetson_nano.toFixed(3)}</Typography>
                <Typography variant="body2">Desktop PC: {ratingDialog.rating.size_score.desktop_pc.toFixed(3)}</Typography>
                <Typography variant="body2">AWS Server: {ratingDialog.rating.size_score.aws_server.toFixed(3)}</Typography>
              </Box>
            </Box>
          ) : (
            <Alert severity="info">No rating data available</Alert>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setRatingDialog({ open: false })}>Close</Button>
        </DialogActions>
      </Dialog>

      {/* Audit Trail Dialog */}
      <Dialog open={auditDialog.open} onClose={() => setAuditDialog({ open: false })} fullWidth maxWidth="md">
        <DialogTitle>
          Audit Trail: {auditDialog.artifact?.name}
        </DialogTitle>
        <DialogContent>
          {auditEntries.length === 0 ? (
            <Alert severity="info">No audit entries found.</Alert>
          ) : (
            <List>
              {auditEntries.map((entry, idx) => (
                <ListItem key={idx}>
                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Chip label={entry.action} size="small" color="primary" />
                        <Typography variant="body2" color="text.secondary">
                          {entry.user.name} {entry.user.is_admin ? '(admin)' : ''}
                        </Typography>
                      </Box>
                    }
                    secondary={new Date(entry.date).toLocaleString()}
                  />
                </ListItem>
              ))}
            </List>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAuditDialog({ open: false })}>Close</Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default ModelDownloadPage;
