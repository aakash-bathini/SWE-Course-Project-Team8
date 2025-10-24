import React from 'react';
import { Container, Typography, Box } from '@mui/material';

interface User {
  username: string;
  permissions: string[];
}

interface ModelSearchPageProps {
  user: User;
}

const ModelSearchPage: React.FC<ModelSearchPageProps> = ({ user }) => {
  return (
    <Container maxWidth="lg">
      <Box sx={{ mt: 4 }}>
        <Typography variant="h4" gutterBottom>
          Search Models
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Search for models in the registry. This feature will be fully implemented in Milestone 4.
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
          User: {user.username} | Permissions: {user.permissions.join(', ')}
        </Typography>
      </Box>
    </Container>
  );
};

export default ModelSearchPage;
