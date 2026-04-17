function entropy = entropy_calculation_fun(T,specie,Ru,MW_specie)
    % entropy_calculation_fun
    %   T       : temperature [K]
    %   specie  : species name string as in janaf_table.txt (e.g. 'O2')
    %   Ru      : universal gas constant
    %   MW_specie : molecular weight of species
    %
    %   Returns: specific entropy [J/(kg*K)]

    R = Ru / MW_specie;

    % ---- Cache JANAF species names so we don't re-scan the whole file ----
    persistent Nsh_cached
    if isempty(Nsh_cached)
        fileIDE = fopen('janaf_table.txt','r');
        if fileIDE == -1
            error('entropy_calculation_fun:CannotOpenJANAF', ...
                  'Could not open janaf_table.txt');
        end
        Msh = textscan(fileIDE,'%s %s %s %s %s','headerlines',0);
        fclose(fileIDE);
        Nsh_cached = Msh{1,1};  % species name column
    end
    % ----------------------------------------------------------------------

    % Find species block (same logic as original code, but using cached names)
    counter = 1;
    for i = 1:4:10000
        if i <= numel(Nsh_cached) && strcmp(Nsh_cached(i), specie) == 1
            break
        end
        counter = counter + 1;
    end

    % If we never found it, fail explicitly
    if counter == 1 && (numel(Nsh_cached) < 1 || strcmp(Nsh_cached(1),specie) ~= 1)
        error('entropy_calculation_fun:SpeciesNotFound', ...
              'Species "%s" not found in janaf_table.txt', specie);
    end

    m = (counter * 4 - 3);

    % Read the coefficient block for this species
    fileID = fopen('janaf_table.txt','r');
    if fileID == -1
        error('entropy_calculation_fun:CannotOpenJANAF', ...
              'Could not open janaf_table.txt');
    end
    Csh = textscan(fileID,'%f %f %f %f %f','headerlines',m);
    fclose(fileID);

    Dsh = zeros(3,5);
    n = 3;
    for k = 1:5
        Ash = Csh{1,k};
        if k == 5
            n = 2;
        end
        for j = 1:n
            Dsh(j,k) = Ash(j,1);
        end
    end

    % High-T and low-T branches: always assign "entropy" in one of these
    if T >= 1000
        entropy = R * ( ...
            Dsh(1,1)*log(T) + ...
            Dsh(1,2)*T      + ...
            Dsh(1,3)*(T^2)/2 + ...
            Dsh(1,4)*(T^3)/3 + ...
            Dsh(1,5)*(T^4)/4 + ...
            Dsh(2,2) );
    else
        entropy = R * ( ...
            Dsh(2,3)*log(T) + ...
            Dsh(2,4)*T      + ...
            Dsh(2,5)*(T^2)/2 + ...
            Dsh(3,1)*(T^3)/3 + ...
            Dsh(3,2)*(T^4)/4 + ...
            Dsh(3,4) );
    end
end