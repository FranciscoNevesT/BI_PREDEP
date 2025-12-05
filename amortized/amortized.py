from amortized.flow import AmortizedFlowModel


def train_amortized_flow(model: AmortizedFlowModel, data_loader, optimizer, epochs):
    model.flow.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in data_loader:
            optimizer.zero_grad()
            x = batch  # Assuming batch is the input data
            log_prob = model.log_prob(x)
            loss = -log_prob.mean()  # Negative log likelihood
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    return model