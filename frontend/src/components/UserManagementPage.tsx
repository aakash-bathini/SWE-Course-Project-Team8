import React, { useEffect, useState } from 'react';
import {
  Container,
  Paper,
  Typography,
  Box,
  TextField,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Checkbox,
  ListItemText,
  Alert,
  CircularProgress,
} from '@mui/material';
import { Delete as DeleteIcon, PersonAdd as PersonAddIcon } from '@mui/icons-material';
import apiService from '../services/apiService';
import { authService } from '../services/authService';

interface User {
  username: string;
  permissions: string[];
}

interface UserManagementPageProps {
  user: User;
  onNotification: (message: string, severity: 'success' | 'error' | 'warning' | 'info') => void;
}

const UserManagementPage: React.FC<UserManagementPageProps> = ({ user, onNotification }) => {
  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(false);
  const [registerDialogOpen, setRegisterDialogOpen] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [selectedUser, setSelectedUser] = useState<string>('');
  const [formData, setFormData] = useState({
    username: '',
    password: '',
    permissions: [] as string[],
  });

  const availablePermissions = ['upload', 'search', 'download', 'admin'];

  const isAdmin = user.permissions.includes('admin');

  const loadUsers = async () => {
    try {
      setLoading(true);
      const list = await apiService.getUsers();
      setUsers(list);
    } catch {
      // Non-fatal: keep any existing users array
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isAdmin) {
      loadUsers();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleRegister = async () => {
    if (!formData.username || !formData.password || formData.permissions.length === 0) {
      onNotification('Please fill in all fields', 'error');
      return;
    }

    try {
      setLoading(true);
      await apiService.registerUser({
        username: formData.username,
        password: formData.password,
        permissions: formData.permissions,
      });
      onNotification('User registered successfully', 'success');
      setRegisterDialogOpen(false);
      setFormData({ username: '', password: '', permissions: [] });
      await loadUsers();
    } catch (error: any) {
      onNotification(error.response?.data?.detail || 'Failed to register user', 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async () => {
    if (!selectedUser) return;

    try {
      setLoading(true);
      await apiService.deleteUser(selectedUser);
      onNotification('User deleted successfully', 'success');
      setDeleteDialogOpen(false);
      setSelectedUser('');
      await loadUsers();
    } catch (error: any) {
      onNotification(error.response?.data?.detail || 'Failed to delete user', 'error');
    } finally {
      setLoading(false);
    }
  };

  if (!isAdmin) {
    return (
      <Container maxWidth="sm">
        <Box sx={{ mt: 6 }}>
          <Typography variant="h5" gutterBottom>
            My Account
          </Typography>
          <Alert severity="info" sx={{ mb: 3 }}>
            You are signed in as <strong>{user.username}</strong>.
          </Alert>
          <Paper sx={{ p: 3 }}>
            <Typography variant="subtitle1" gutterBottom>
              Delete Account
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              This action permanently deletes your account. You will be signed out immediately.
            </Typography>
            <Button
              color="error"
              variant="contained"
              onClick={() => {
                setSelectedUser(user.username);
                setDeleteDialogOpen(true);
              }}
              disabled={loading}
            >
              {loading ? <CircularProgress size={24} /> : 'Delete My Account'}
            </Button>
          </Paper>
        </Box>

        {/* Delete User Dialog (self-delete) */}
        <Dialog open={deleteDialogOpen} onClose={() => setDeleteDialogOpen(false)}>
          <DialogTitle>Delete Account</DialogTitle>
          <DialogContent>
            <Typography>
              Are you sure you want to delete your account <strong>{selectedUser}</strong>?
            </Typography>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setDeleteDialogOpen(false)}>Cancel</Button>
            <Button
              onClick={async () => {
                try {
                  setLoading(true);
                  await apiService.deleteUser(selectedUser);
                  onNotification('Account deleted. Signing out…', 'success');
                  localStorage.removeItem('token');
                  window.location.href = '/login';
                } catch (error: any) {
                  onNotification(error.response?.data?.detail || 'Failed to delete account', 'error');
                } finally {
                  setLoading(false);
                }
              }}
              color="error"
              variant="contained"
              disabled={loading}
            >
              {loading ? <CircularProgress size={24} /> : 'Delete'}
            </Button>
          </DialogActions>
        </Dialog>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg">
      <Box sx={{ mt: 4 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Typography variant="h4">User Management</Typography>
          <Button
            variant="contained"
            startIcon={<PersonAddIcon />}
            onClick={() => setRegisterDialogOpen(true)}
          >
            Register New User
          </Button>
        </Box>

        <Paper>
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Username</TableCell>
                  <TableCell>Permissions</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {users.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={3} align="center">
                      <Typography variant="body2" color="text.secondary">
                        {loading ? 'Loading users...' : 'No users found'}
                      </Typography>
                    </TableCell>
                  </TableRow>
                ) : (
                  users.map((u) => (
                    <TableRow key={u.username}>
                      <TableCell>{u.username}</TableCell>
                      <TableCell>
                        {u.permissions.map((p) => (
                          <Chip key={p} label={p} size="small" sx={{ mr: 0.5 }} />
                        ))}
                      </TableCell>
                      <TableCell>
                        {u.username === 'ece30861defaultadminuser' ? (
                          <Chip label="Default Admin" color="primary" size="small" />
                        ) : (
                          <IconButton
                            aria-label="delete user"
                            color="error"
                            onClick={() => {
                              setSelectedUser(u.username);
                              setDeleteDialogOpen(true);
                            }}
                          >
                            <DeleteIcon />
                          </IconButton>
                        )}
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>

        {/* Register User Dialog */}
        <Dialog open={registerDialogOpen} onClose={() => setRegisterDialogOpen(false)} maxWidth="sm" fullWidth>
          <DialogTitle>Register New User</DialogTitle>
          <DialogContent>
            <Box sx={{ pt: 2 }}>
              <TextField
                fullWidth
                label="Username"
                value={formData.username}
                onChange={(e) => setFormData({ ...formData, username: e.target.value })}
                margin="normal"
                required
              />
              <TextField
                fullWidth
                label="Password"
                type="password"
                value={formData.password}
                onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                margin="normal"
                required
              />
              <FormControl fullWidth margin="normal">
                <InputLabel>Permissions</InputLabel>
                <Select
                  multiple
                  value={formData.permissions}
                  onChange={(e) => setFormData({ ...formData, permissions: e.target.value as string[] })}
                  renderValue={(selected) => (
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                      {(selected as string[]).map((value) => (
                        <Chip key={value} label={value} size="small" />
                      ))}
                    </Box>
                  )}
                >
                  {availablePermissions.map((perm) => (
                    <MenuItem key={perm} value={perm}>
                      <Checkbox checked={formData.permissions.indexOf(perm) > -1} />
                      <ListItemText primary={perm} />
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Box>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setRegisterDialogOpen(false)}>Cancel</Button>
            <Button onClick={handleRegister} variant="contained" disabled={loading}>
              {loading ? <CircularProgress size={24} /> : 'Register'}
            </Button>
          </DialogActions>
        </Dialog>

        {/* Delete User Dialog */}
        <Dialog open={deleteDialogOpen} onClose={() => setDeleteDialogOpen(false)}>
          <DialogTitle>Delete User</DialogTitle>
          <DialogContent>
            <Typography>
              Are you sure you want to delete user <strong>{selectedUser}</strong>?
            </Typography>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setDeleteDialogOpen(false)}>Cancel</Button>
            <Button onClick={handleDelete} color="error" variant="contained" disabled={loading}>
              {loading ? <CircularProgress size={24} /> : 'Delete'}
            </Button>
          </DialogActions>
        </Dialog>
      </Box>
    </Container>
  );
};

export default UserManagementPage;

