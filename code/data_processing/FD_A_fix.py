import torch

FDA_train = torch.load('datasets\\FD_A\\train.pt')
FDA_val = torch.load('datasets\\FD_A\\val.pt')
FDA_test = torch.load('datasets\\FD_A\\test.pt')
FDB_train = torch.load('datasets\\FD_B\\train.pt')
FDB_val = torch.load('datasets\\FD_B\\val.pt')
FDB_test = torch.load('datasets\\FD_B\\test.pt')


if not type(FDA_train['samples']) == torch.Tensor:
    FDA_torch_train = torch.tensor(FDA_train['samples'])
    FDA_torch_val = torch.tensor(FDA_val['samples'])
    FDA_torch_test = torch.tensor(FDA_test['samples'])
    FDB_torch_train = torch.tensor(FDB_train['samples'])
    FDB_torch_val = torch.tensor(FDB_val['samples'])
    FDB_torch_test = torch.tensor(FDB_test['samples'])

    FDA_train['samples'] = FDA_torch_train
    FDA_val['samples'] = FDA_torch_val
    FDA_test['samples'] = FDA_torch_test
    FDB_train['samples'] = FDB_torch_train
    FDB_val['samples'] = FDB_torch_val
    FDB_test['samples'] = FDB_torch_test

    torch.save(FDA_train, 'datasets\\FD_A\\train.pt')
    torch.save(FDA_val, 'datasets\\FD_A\\val.pt')
    torch.save(FDA_test, 'datasets\\FD_A\\test.pt')
    torch.save(FDB_train, 'datasets\\FD_B\\train.pt')
    torch.save(FDB_val, 'datasets\\FD_B\\val.pt')
    torch.save(FDB_test, 'datasets\\FD_B\\test.pt')

