import React from 'react';
import { Container, Typography, Box } from '@mui/material';

interface User {
  username: string;
  permissions: string[];
}

interface ModelDownloadPageProps {
  user: User;
}

const ModelDownloadPage: React.FC<ModelDownloadPageProps> = ({ user }) => {
  return (
    <Container maxWidth="lg">
      <Box sx={{ mt: 4 }}>
        <Typography variant="h4" gutterBottom>
          Download Models
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Download models from the registry. This feature will be fully implemented in Milestone 2.
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
          User: {user.username} | Permissions: {user.permissions.join(', ')}
        </Typography>
      </Box>
    </Container>
  );
};

export default ModelDownloadPage;
