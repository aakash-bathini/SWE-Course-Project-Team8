import React, { useEffect, useState } from 'react';
import { Container, Typography, Box, TextField, Button, Stack, MenuItem, ToggleButtonGroup, ToggleButton, Paper, List, ListItem, ListItemText, Alert } from '@mui/material';
import { apiService, type ArtifactMetadata } from '../services/apiService';

interface User {
  username: string;
  permissions: string[];
}

interface ModelSearchPageProps {
  user: User;
}

const ModelSearchPage: React.FC<ModelSearchPageProps> = ({ user }) => {
  const [mode, setMode] = useState<'name' | 'regex'>('name');
  const [artifactType, setArtifactType] = useState<'model' | 'dataset' | 'code' | 'any'>('any');
  const [query, setQuery] = useState('');
  const [items, setItems] = useState<ArtifactMetadata[]>([]);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const canSearch = user.permissions.includes('search') || user.permissions.includes('download');

  const runSearch = async () => {
    if (!canSearch) return;
    setError('');
    setLoading(true);
    try {
      let results: ArtifactMetadata[] = [];
      if (mode === 'name') {
        const qtrim = query.trim();
        if (!qtrim || qtrim === '*') {
          const q = artifactType === 'any' ? [{ name: '*' }] : [{ name: '*', types: [artifactType] } as any];
          results = await apiService.listArtifacts(q, '0');
        } else {
          results = await apiService.searchByName(qtrim);
          if (artifactType !== 'any') {
            results = results.filter((m) => m.type === artifactType);
          }
        }
      } else {
        const rx = query.trim() || '.*';
        results = await apiService.searchByRegex(rx);
        if (artifactType !== 'any') {
          results = results.filter((m) => m.type === artifactType);
        }
      }
      setItems(results);
    } catch (e: any) {
      setError(e?.message || 'Search failed');
    } finally {
      setLoading(false);
    }
  };

  // Auto-run a wildcard search on initial load to show existing artifacts
  useEffect(() => {
    runSearch();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <Container maxWidth="lg">
      <Box sx={{ mt: 4 }}>
        <Typography variant="h4" gutterBottom>
          Search Models
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          User: {user.username} | Permissions: {user.permissions.join(', ')}
        </Typography>
        {!canSearch && <Alert severity="warning" sx={{ mb: 2 }}>Your account lacks 'search' permission.</Alert>}
        <Stack spacing={2} direction={{ xs: 'column', sm: 'row' }} sx={{ mb: 2 }}>
          <ToggleButtonGroup
            color="primary"
            value={mode}
            exclusive
            onChange={(_, v) => v && setMode(v)}
          >
            <ToggleButton value="name">By Name</ToggleButton>
            <ToggleButton value="regex">By Regex</ToggleButton>
          </ToggleButtonGroup>
          <TextField
            label={mode === 'name' ? 'Name (use * for all)' : 'Regex (e.g., .*bert.*)'}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            fullWidth
          />
          <TextField label="Type" select value={artifactType} onChange={(e) => setArtifactType(e.target.value as any)} sx={{ minWidth: 160 }}>
            <MenuItem value="any">any</MenuItem>
            <MenuItem value="model">model</MenuItem>
            <MenuItem value="dataset">dataset</MenuItem>
            <MenuItem value="code">code</MenuItem>
          </TextField>
          <Button variant="contained" onClick={runSearch} disabled={loading || !canSearch}>
            {loading ? 'Searching...' : 'Search'}
          </Button>
        </Stack>
        {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
        <Paper variant="outlined">
          <List dense>
            {items.map((m) => (
              <ListItem key={m.id}>
                <ListItemText primary={`${m.name} (${m.type})`} secondary={`id: ${m.id}`} />
              </ListItem>
            ))}
            {!items.length && <ListItem><ListItemText primary="No results" /></ListItem>}
          </List>
        </Paper>
      </Box>
    </Container>
  );
};

export default ModelSearchPage;
