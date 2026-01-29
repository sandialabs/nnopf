import matplotlib.pyplot as plt

# CASE 14
val_losses_14=[0.00015417946820283154, 0.0001322127765888581, 3.69855829012522e-05, 3.0793713212915465e-05, 1.502491047934503e-05, 4.195269394282756e-05, 9.668274786539163e-06, 1.3846912299110651e-05, 1.8080753143294715e-05, 1.584633429047244e-05]
train_losses_14=[0.004645926479523513, 0.00012281316212280486, 6.713179449063857e-05, 4.048837701685881e-05, 2.955690776094276e-05, 2.592018926853362e-05, 2.4228527049756478e-05, 2.1690395856366093e-05, 1.792135348578254e-05, 1.6424776699118918e-05]
# CASE 30
val_losses_30=[0.00014400374212224657, 0.00010955031581640166, 4.068171419930877e-05, 3.3021869852139694e-05, 1.696363909559295e-05, 1.8652498047231346e-05, 1.715659464783433e-05, 1.1297335116372173e-05, 1.8149009771756634e-05, 1.0790745185052704e-05]
train_losses_30=[0.0035375347627834546, 0.00012786486080720486, 8.136376922350422e-05, 4.955343406773291e-05, 3.4825986888672894e-05, 2.1994510530358067e-05, 1.7495632199808353e-05, 1.640987776378703e-05, 1.2493916754389526e-05, 1.0533592635637332e-05]

# Create a figure
plt.figure()

# Add traces for training and validation losses
plt.plot(range(len(train_losses_14)), train_losses_14, label='Training MSE (Case 14)', linestyle='-')
plt.plot(range(len(val_losses_14)), val_losses_14, label='Validation MSE (Case 14)', linestyle='--')
plt.plot(range(len(train_losses_30)), train_losses_30, label='Training MSE (Case 30)', linestyle=':')
plt.plot(range(len(val_losses_30)), val_losses_30, label='Validation MSE (Case 30)', linestyle='-.')

# Update layout
plt.title(f'Training Curves for pglib_opf Cases 14 and 30')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error (MSE)')
plt.yscale('log')
plt.legend()

# Show grid lines
plt.grid(True)

# Save the figure as a PNG file
plt.savefig(f'TRAINFIG.png')

# Show the figure
plt.show()
