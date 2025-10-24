import React from 'react';
import { Container, Typography, Box } from '@mui/material';

interface User {
  username: string;
  permissions: string[];
}

interface ModelUploadPageProps {
  user: User;
  onNotification: (message: string, severity: 'success' | 'error' | 'warning' | 'info') => void;
}

const ModelUploadPage: React.FC<ModelUploadPageProps> = ({ user, onNotification }) => {
  return (
    <Container maxWidth="lg">
      <Box sx={{ mt: 4 }}>
        <Typography variant="h4" gutterBottom>
          Upload Model
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Upload a new model to the registry. This feature will be fully implemented in Milestone 2.
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
          User: {user.username} | Permissions: {user.permissions.join(', ')}
        </Typography>
      </Box>
    </Container>
  );
};

export default ModelUploadPage;
