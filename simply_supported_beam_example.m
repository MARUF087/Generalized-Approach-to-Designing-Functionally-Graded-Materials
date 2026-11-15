clear all; close all; clc;
tic
%% 2D Left Half Simply Supported Beam FEM with Orthotropic Properties and Optimization

% Parameters
L = 50;          % Length (mm)
h = 10;           % Height (mm)
P = -0.1;         % Total load (KN, downward)
thickness = 1;    % Thickness (mm)
I = h^3/12;        % Moment of inertia (mm^4)

% Initial orthotropic material properties (stiffness constants, GPa)
C11 = 283.5271;
C22 = 283.5271;
C12 = 117.0405;
C66 = 82.5151;

% Mesh generation
nx = 250; ny = 50;
[x, y] = meshgrid(linspace(0,L,nx+1), linspace(-h/2,h/2,ny+1));
nodes = [x(:), y(:)];
elements = delaunay(nodes(:,1), nodes(:,2));        % Equivalent CPS3 element of ABAQUS
nNodes = size(nodes,1);
nElem = size(elements,1);

% Function to run FEM simulation with uniform orthotropic properties
function [stress, strain, UX, UY] = runFEM(nodes, elements, C11, C22, C12, C66, L, P, thickness, nNodes, nElem, nx)
    % Material matrix (Orthotropic plane stress)
    D = [C11 C12 0;
         C12 C22 0;
         0   0   C66];
    
    % Assemble global stiffness matrix
    nDOF = 2*nNodes; % 2 DOFs per node (ux, uy)
    K = sparse(nDOF, nDOF);
    
    for e = 1:nElem
        nodeIDs = elements(e,:);
        elemNodes = nodes(nodeIDs,:);
        [Ke, ~] = tri3K_2DOF(elemNodes, D, thickness);
        dofs = reshape([2*nodeIDs-1; 2*nodeIDs], [], 1);
        K(dofs, dofs) = K(dofs, dofs) + Ke;
    end
    
    % Boundary conditions (simply supported left half: uy=0 at x=0, ux=0 at x=L)
    leftNodes = find(abs(nodes(:,1)) < 1e-6); % x=0
    rightNodes = find(abs(nodes(:,1) - L) < L/(2*nx)); % x=L
    fixedDOFs = sort([2*leftNodes; 2*rightNodes-1]); % uy at x=0, ux at x=L
    
    % Applied load (distributed at right end, x=L)
    force = zeros(nDOF,1);
    force(2*rightNodes) = P / length(rightNodes); % Apply load to uy DOFs
    
    % Solve
    freeDOFs = setdiff(1:nDOF, fixedDOFs);
    U = zeros(nDOF,1);
    U(freeDOFs) = K(freeDOFs,freeDOFs) \ force(freeDOFs);
    
    % Extract displacements
    UX = U(1:2:end); % x-displacement
    UY = U(2:2:end); % y-displacement
    
    % Compute stress and strain
    stress = zeros(nElem,3);
    strain = zeros(nElem,3);
    
    for e = 1:nElem
        nodeIDs = elements(e,:);
        elemNodes = nodes(nodeIDs,:);
        [~, B] = tri3K_2DOF(elemNodes, D, thickness);
        dofs = reshape([2*nodeIDs-1; 2*nodeIDs], [], 1);
        u_e = U(dofs);
        strain(e,:) = (B * u_e)';
        stress(e,:) = (D * strain(e,:)')';
    end
end

% Function to run FEM with element-specific optimized properties
function [stress, strain, UX, UY] = runFEMOpt(nodes, elements, C11_opt, C22_opt, C12_opt, C66_opt, L, P, thickness, nNodes, nElem, nx)
    % Assemble global stiffness matrix
    nDOF = 2*nNodes; % 2 DOFs per node (ux, uy)
    K = sparse(nDOF, nDOF);
    
    for e = 1:nElem
        % Element-specific material matrix
        D = [C11_opt(e) C12_opt(e) 0;
             C12_opt(e) C22_opt(e) 0;
             0          0          C66_opt(e)];
        
        nodeIDs = elements(e,:);
        elemNodes = nodes(nodeIDs,:);
        [Ke, ~] = tri3K_2DOF(elemNodes, D, thickness);
        dofs = reshape([2*nodeIDs-1; 2*nodeIDs], [], 1);
        K(dofs, dofs) = K(dofs, dofs) + Ke;
    end
    
    % Boundary conditions
    leftNodes = find(abs(nodes(:,1)) < 1e-6); % x=0
    rightNodes = find(abs(nodes(:,1) - L) < L/(2*nx)); % x=L
    fixedDOFs = sort([2*leftNodes; 2*rightNodes-1]); % uy at x=0, ux at x=L
    
    % Applied load (distributed at right end, x=L)
    force = zeros(nDOF,1);
    force(2*rightNodes) = P / length(rightNodes); % Apply load to uy DOFs
    
    % Solve
    freeDOFs = setdiff(1:nDOF, fixedDOFs);
    U = zeros(nDOF,1);
    U(freeDOFs) = K(freeDOFs,freeDOFs) \ force(freeDOFs);
    
    % Extract displacements
    UX = U(1:2:end);
    UY = U(2:2:end);
    
    % Compute stress and strain
    stress = zeros(nElem,3);
    strain = zeros(nElem,3);
    
    for e = 1:nElem
        D = [C11_opt(e) C12_opt(e) 0;
             C12_opt(e) C22_opt(e) 0;
             0          0          C66_opt(e)];
        
        nodeIDs = elements(e,:);
        elemNodes = nodes(nodeIDs,:);
        [~, B] = tri3K_2DOF(elemNodes, D, thickness);
        dofs = reshape([2*nodeIDs-1; 2*nodeIDs], [], 1);
        u_e = U(dofs);
        strain(e,:) = (B * u_e)';
        stress(e,:) = (D * strain(e,:)')';
    end
end

% Initialize storage for results across iterations
nIterations = 4;
stress_all = cell(nIterations, 1);
strain_all = cell(nIterations, 1);
UX_all = cell(nIterations, 1);
UY_all = cell(nIterations, 1);
C11_opt_all = cell(nIterations-1, 1);
C22_opt_all = cell(nIterations-1, 1);
C12_opt_all = cell(nIterations-1, 1);
C66_opt_all = cell(nIterations-1, 1);

% Initial FEM run (Iteration 1)
[stress_all{1}, strain_all{1}, UX_all{1}, UY_all{1}] = runFEM(nodes, elements, C11, C22, C12, C66, L, P, thickness, nNodes, nElem, nx);

% Iterative optimization
for iter = 1:nIterations-1
    % Initialize optimized properties for this iteration
    C11_opt = zeros(nElem,1);
    C22_opt = zeros(nElem,1);
    C12_opt = zeros(nElem,1);
    C66_opt = zeros(nElem,1);
    
    % Use stress from previous iteration
    stress_prev = stress_all{iter};
    
    % Optimize for each element
    for e = 1:nElem
        % Stress components for element
        sigma_11 = stress_prev(e,1);
        sigma_22 = stress_prev(e,2);
        tau_12 = stress_prev(e,3);
        
        % Objective function
        f = @(x) ((- sigma_11^2 *x(2) + 2 * sigma_11 * sigma_22 * x(3) - sigma_22^2 * x(1)) * (1 / (x(3)^2 - x(1)*x(2))) + tau_12^2/x(4));
        
        % Initial guess
        if iter == 1
            x0 = [283.5271; 283.5271; 117.0405; 82.5151];
        else
            x0 = [C11_opt_all{iter-1}(e); C22_opt_all{iter-1}(e); C12_opt_all{iter-1}(e); C66_opt_all{iter-1}(e)];
        end
        
        % Constraints
        A = [0.27685, 0.27685, 0.92017, 0;
             -0.15576, -0.15066, 0.97624, 0;
             -0.63256, -0.05766, -0.77236, 0;
             -0.64474, 0.40416, -0.64882, 0;
             0.32903, 0.80507, -0.49356, 0;
             -0.62874, 0.45171, -0.63297, 0;
             0.39886, 0.51041, 0.76184, 0;
             0.80507, 0.32903, -0.49356, 0;
             0.51041, 0.39886, 0.76184, 0;
             0.42855, 0.42855, 0.79541, 0;
             0.45171, -0.62874, -0.63297, 0;
             -0.05766, -0.63256, -0.77236, 0;
             -0.15066, -0.15576, 0.97624, 0;
             0.27685, 0.27685, 0.92017, 0;
             0.40416, -0.64474, -0.64882, 0;
             -0.15321, -0.15321, 0.97625, 0];
        b = [270.44284; 47.22389; -278.51010; -134.13403; 285.55942; -113.88138;
             351.93696; 285.55942; 351.93696; 339.95893; -113.88138; -278.51010;
             47.22389; 270.44284; -134.13403; 47.22683];
        Aeq = [0, 0, 1.0018, -1];
        beq = [34.736];
        
        % Optimize
                % options = optimoptions('fmincon', 'Display', 'off');

     options = optimoptions('fmincon', ...
    'Display', 'off', ...
    'OptimalityTolerance', 1e-9, ...
    'StepTolerance', 1e-10, ...
    'ConstraintTolerance', 1e-9, ...
    'MaxIterations', 2000, ...
    'MaxFunctionEvaluations', 1e5, ...
    'Algorithm', 'interior-point');

        [x_opt, ~] = fmincon(f, x0, A, b, Aeq, beq, [], [], [], options);
        
        % Extract stiffness constants
        C11_opt(e) = x_opt(1);
        C22_opt(e) = x_opt(2);
        C12_opt(e) = x_opt(3);
        C66_opt(e) = x_opt(4);
    end
    
    % Store optimized properties
    C11_opt_all{iter} = C11_opt;
    C22_opt_all{iter} = C22_opt;
    C12_opt_all{iter} = C12_opt;
    C66_opt_all{iter} = C66_opt;
    
    % Run FEM with optimized properties
    [stress_all{iter+1}, strain_all{iter+1}, UX_all{iter+1}, UY_all{iter+1}] = runFEMOpt(nodes, elements, C11_opt, C22_opt, C12_opt, C66_opt, L, P, thickness, nNodes, nElem, nx);
end

% Save optimized properties from final iteration to Excel
stiffMatrix = [C11_opt_all{end}, C22_opt_all{end}, C12_opt_all{end}, C66_opt_all{end}];
header = {'C11 (GPa)', 'C22 (GPa)', 'C12 (GPa)', 'C66 (GPa)'};
T = array2table(stiffMatrix, 'VariableNames', header);
writetable(T, 'Optimized_Stiffness_Constants_Final.xlsx');

% Plot results
% Stress
stressLabels = {'$\sigma_{xx}$ (GPa)', '$\sigma_{yy}$ (GPa)', '$\tau_{xy}$ (GPa)'};
for i = 1:3
    figure
    for iter = 1:nIterations
        subplot(2,2,iter)
        patch('Faces',elements, 'Vertices',nodes, ...
              'FaceVertexCData',stress_all{iter}(:,i), 'FaceColor','flat', 'EdgeColor','none')
        if iter == 1
            title(['Initial ', stressLabels{i}], 'Interpreter','latex', 'FontSize',14)
        else
            title(['Iter ', num2str(iter), ' ', stressLabels{i}], 'Interpreter','latex', 'FontSize',14)
        end
        colorbar('southoutside'); axis equal tight; colormap jet
        set(gca, 'FontSize',12)
    end
end

% Strain
strainLabels = {'$\epsilon_{xx}$', '$\epsilon_{yy}$', '$\gamma_{xy}$'};
for i = 1:3
    figure
    for iter = 1:nIterations
        subplot(2,2,iter)
        patch('Faces',elements, 'Vertices',nodes, ...
              'FaceVertexCData',strain_all{iter}(:,i), 'FaceColor','flat', 'EdgeColor','none')
        if iter == 1
            title(['Initial ', strainLabels{i}], 'Interpreter','latex', 'FontSize',14)
        else
            title(['Iter ', num2str(iter), ' ', strainLabels{i}], 'Interpreter','latex', 'FontSize',14)
        end
        colorbar('southoutside'); axis equal tight; colormap jet
        set(gca, 'FontSize',12)
    end
end

% Vertical Deflection (uy)
figure
for iter = 1:nIterations
    subplot(2,2,iter)
    patch('Faces',elements, 'Vertices',nodes, ...
          'FaceVertexCData',UY_all{iter}, 'FaceColor','interp', 'EdgeColor','none')
    if iter == 1
        title('Initial Vertical Deflection $u_y$ (mm)', 'Interpreter','latex', 'FontSize',14)
    else
        title(['Iter ', num2str(iter), ' Vertical Deflection $u_y$ (mm)'], 'Interpreter','latex', 'FontSize',14)
    end
    xlabel('X (mm)'); ylabel('Y (mm)')
    colorbar('southoutside'); axis equal tight; colormap jet
    set(gca, 'FontSize',12)
end

% Horizontal Deflection (ux)
figure
for iter = 1:nIterations
    subplot(2,2,iter)
    patch('Faces',elements, 'Vertices',nodes, ...
          'FaceVertexCData',UX_all{iter}, 'FaceColor','interp', 'EdgeColor','none')
    if iter == 1
        title('Initial Horizontal Deflection $u_x$ (mm)', 'Interpreter','latex', 'FontSize',14)
    else
        title(['Iter ', num2str(iter), ' Horizontal Deflection $u_x$ (mm)'], 'Interpreter','latex', 'FontSize',14)
    end
    xlabel('X (mm)'); ylabel('Y (mm)')
    colorbar('southoutside'); axis equal tight; colormap jet
    set(gca, 'FontSize',12)
end

% Optimized stiffness constants
stiffLabels = {'$C_{11}$ (GPa)', '$C_{22}$ (GPa)', '$C_{12}$ (GPa)', '$C_{66}$ (GPa)'};
C11_initial = C11 * ones(nElem, 1);
C22_initial = C22 * ones(nElem, 1);
C12_initial = C12 * ones(nElem, 1);
C66_initial = C66 * ones(nElem, 1);
stiffData = {{C11_initial, C11_opt_all{1}, C11_opt_all{2}, C11_opt_all{3}}, ...
             {C22_initial, C22_opt_all{1}, C22_opt_all{2}, C22_opt_all{3}}, ...
             {C12_initial, C12_opt_all{1}, C12_opt_all{2}, C12_opt_all{3}}, ...
             {C66_initial, C66_opt_all{1}, C66_opt_all{2}, C66_opt_all{3}}};

for i = 1:4
    figure('Name',['Optimized ', stiffLabels{i}],'Units','normalized','OuterPosition',[0 0 1 1])
    for iter = 1:nIterations
        subplot(2,2,iter)
        patch('Faces', elements, 'Vertices', nodes, ...
              'FaceVertexCData', stiffData{i}{iter}, 'FaceColor', 'flat', 'EdgeColor', 'none')
        if iter == 1
            title(['Initial ', stiffLabels{i}], 'Interpreter', 'latex', 'FontSize', 14)
        else
            title(['Iter ', num2str(iter), ' ', stiffLabels{i}], 'Interpreter', 'latex', 'FontSize', 14)
        end
        xlabel('X (mm)'); ylabel('Y (mm)')
        colorbar('southoutside'); axis equal tight; colormap jet
        set(gca, 'FontSize', 12)
    end
end

% Plot Maximum Vertical Deflection vs Iteration (with LaTeX fonts)
maxDeflections = zeros(nIterations, 1);
for iter = 1:nIterations
    maxDeflections(iter) = min(UY_all{iter});  % Negative downward deflection
end

figure
plot(1:nIterations, maxDeflections, 'o-k', 'LineWidth', 2, 'MarkerSize', 8)
xlabel('\textbf{Iteration}', 'Interpreter', 'latex')
ylabel('\textbf{Maximum Vertical Deflection} $u_y$ \textbf{(mm)}', 'Interpreter', 'latex')
title('\textbf{Maximum Vertical Deflection vs Iteration}', 'Interpreter', 'latex')
grid on
set(gca, 'FontSize', 12)
% Plot Maximum Horizontal Deflection vs Iteration (with LaTeX fonts)
maxDeflectionsX = zeros(nIterations, 1);
for iter = 1:nIterations
    maxDeflectionsX(iter) = max(UX_all{iter});  % Positive rightward deflection
end

figure
plot(1:nIterations, maxDeflectionsX, 'o-k', 'LineWidth', 2, 'MarkerSize', 8)
xlabel('\textbf{Iteration}', 'Interpreter', 'latex')
ylabel('\textbf{Maximum Horizontal Deflection} $u_x$ \textbf{(mm)}', 'Interpreter', 'latex')
title('\textbf{Maximum Horizontal Deflection vs Iteration}', 'Interpreter', 'latex')
grid on
set(gca, 'FontSize', 12)


%% Element stiffness function (2 DOFs: ux, uy)
function [Ke, B] = tri3K_2DOF(nodes, D, t)
    % Computes stiffness matrix and strain-displacement matrix for a 3-node
    % triangular element with 2 DOFs per node (ux, uy) under plane stress
    x = nodes(:,1); y = nodes(:,2);
    A = polyarea(x, y); % Element area
    beta = [y(2)-y(3); y(3)-y(1); y(1)-y(2)];
    gamma = [x(3)-x(2); x(1)-x(3); x(2)-x(1)];
    N_x = beta/(2*A); % Shape function derivatives
    N_y = gamma/(2*A);
    
    % Strain-displacement matrix B (3x6)
    B = zeros(3,6);
    for i = 1:3
        B(1,2*i-1) = N_x(i); % epsilon_xx = du_x/dx
        B(2,2*i) = N_y(i);   % epsilon_yy = du_y/dy
        B(3,2*i-1) = N_y(i); % gamma_xy = du_x/dy + du_y/dx
        B(3,2*i) = N_x(i);
    end
    Ke = (B' * D * B) * A * t; % Element stiffness matrix
end
toc